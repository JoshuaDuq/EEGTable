// Package eegtable is the only place the TUI talks to Python: one subprocess
// boundary around `eegtable preprocess`, with no terminal dependency.
package eegtable

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
)

type Client struct {
	Binary string
	Config string
	Jobs   int
}

type CommandError struct {
	Args   []string
	Code   int
	Stderr string
}

func (e *CommandError) Error() string {
	return fmt.Sprintf("eegtable %s: exit %d\n%s", strings.Join(e.Args, " "), e.Code, e.Stderr)
}

func Locate() (string, error) {
	if path := os.Getenv("EEGTABLE"); path != "" {
		return path, nil
	}
	path, err := exec.LookPath("eegtable")
	if err != nil {
		return "", errors.New("eegtable not found on PATH: set EEGTABLE to the executable, " +
			"or pip install 'eegtable[preprocessing]' into the active environment")
	}
	return path, nil
}

func (c Client) Status(ctx context.Context) (Status, error) {
	return decode[Status](c.run(ctx, "preprocess", "status", c.Config, "--json"))
}

func (c Client) Inspect(ctx context.Context, recording, stage string) (Gate, error) {
	return decode[Gate](c.run(ctx, "preprocess", "inspect", c.Config, stage, "--recording", recording, "--json"))
}

// Review hands Python a JSON decision file; JSON is valid YAML, and every
// check on it (parent_id, fit_id, nulls) stays in Python.
func (c Client) Review(ctx context.Context, recording, target string, decision Decision) error {
	file, err := os.CreateTemp("", "eegtable-decision-*.json")
	if err != nil {
		return err
	}
	defer os.Remove(file.Name())
	if err := json.NewEncoder(file).Encode(decision); err != nil {
		file.Close()
		return err
	}
	if err := file.Close(); err != nil {
		return err
	}
	_, err = c.run(ctx, "preprocess", "review", c.Config, target, "--recording", recording, "--decisions", file.Name())
	return err
}

func (c Client) Reset(ctx context.Context, recording, stage string) (string, error) {
	out, err := c.run(ctx, "preprocess", "reset", c.Config, "--from", stage, "--recording", recording)
	return strings.TrimSpace(string(out)), err
}

func (c Client) Viewer(recording, stage string) *exec.Cmd {
	return exec.Command(c.Binary, "preprocess", "inspect", c.Config, stage, "--recording", recording)
}

func (c Client) run(ctx context.Context, args ...string) ([]byte, error) {
	cmd := exec.CommandContext(ctx, c.Binary, args...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	cmd.Env = append(os.Environ(), "NO_COLOR=1")
	if err := cmd.Run(); err != nil {
		var exit *exec.ExitError
		if !errors.As(err, &exit) {
			return nil, fmt.Errorf("eegtable %s: %w", strings.Join(args, " "), err)
		}
		return nil, &CommandError{args, exit.ExitCode(), strings.TrimSpace(stderr.String())}
	}
	return stdout.Bytes(), nil
}

func decode[T any](data []byte, err error) (T, error) {
	var value T
	if err != nil {
		return value, err
	}
	if err := json.Unmarshal(data, &value); err != nil {
		return value, fmt.Errorf("eegtable: unreadable JSON output: %w", err)
	}
	return value, nil
}

// Runner is a live `run`: events as Python emits them, then one Result.
type Runner interface {
	Events() <-chan Event
	Done() <-chan Result
	Stop()
}

type Process struct {
	events <-chan Event
	done   <-chan Result
	cmd    *exec.Cmd
}

func (p *Process) Events() <-chan Event { return p.events }
func (p *Process) Done() <-chan Result  { return p.done }

func (c Client) Run(recording string) (Runner, error) {
	args := []string{"preprocess", "run", c.Config}
	if recording != "" {
		args = append(args, "--recording", recording)
	}
	args = append(args, "--n-jobs", strconv.Itoa(c.Jobs))
	args = append(args, "--progress-json")
	cmd := exec.Command(c.Binary, args...)
	cmd.Env = append(os.Environ(), "NO_COLOR=1", "PYTHONUNBUFFERED=1")
	newGroup(cmd)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("eegtable %s: %w", strings.Join(args, " "), err)
	}
	events, done := make(chan Event, 64), make(chan Result, 1)
	var tail []string
	var pipes sync.WaitGroup
	pipes.Add(2)
	go func() {
		defer pipes.Done()
		for line := range lines(stdout) {
			var event Event
			if json.Unmarshal([]byte(line), &event) == nil && event.Event != "" {
				events <- event
				continue
			}
			// Not an event: a worker process or a C extension wrote to the real
			// stdout, which Python's redirect never reaches. Dropping it hides output.
			if line = strings.TrimSpace(line); line != "" {
				events <- Event{Event: "stdout", Message: line}
			}
		}
	}()
	go func() {
		defer pipes.Done()
		for line := range lines(stderr) {
			if line = strings.TrimSpace(line); line == "" {
				continue
			}
			events <- Event{Event: "stderr", Message: line}
			if tail = append(tail, line); len(tail) > stderrTail {
				tail = tail[1:]
			}
		}
	}()
	go func() {
		pipes.Wait()
		close(events)
		result := Result{Stderr: strings.Join(tail, "\n")}
		if err := cmd.Wait(); err != nil {
			var exit *exec.ExitError
			if errors.As(err, &exit) {
				result.Code = exit.ExitCode()
			} else {
				result.Code = -1
			}
			if result.Code == -1 {
				result.Stderr = strings.TrimSpace(result.Stderr + "\n" + err.Error())
			}
		}
		done <- result
	}()
	return &Process{events: events, done: done, cmd: cmd}, nil
}

const stderrTail = 20

const lineCap = 64 * 1024

// lines splits a pipe on newlines and on carriage returns, so a progress bar
// redrawing itself in place arrives as lines instead of one that never ends.
// A line past lineCap is cut rather than ending the read: a reader that stops
// leaves the pipe to fill, and Python then blocks forever on its next write.
func lines(reader io.Reader) <-chan string {
	out := make(chan string)
	go func() {
		defer close(out)
		buffered := bufio.NewReaderSize(reader, 64*1024)
		line := make([]byte, 0, 1024)
		for {
			b, err := buffered.ReadByte()
			if err != nil {
				if len(line) > 0 {
					out <- string(line)
				}
				return
			}
			if b == '\n' || b == '\r' || len(line) >= lineCap {
				if len(line) > 0 {
					out <- string(line)
					line = line[:0]
				}
				if b == '\n' || b == '\r' {
					continue
				}
			}
			line = append(line, b)
		}
	}()
	return out
}

func (p *Process) Stop() {
	if p.cmd != nil && p.cmd.Process != nil {
		stopGroup(p.cmd)
	}
}

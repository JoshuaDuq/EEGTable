package eegtable

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// fake writes an eegtable stand-in that records its argv, replays fixtures by
// subcommand, copies the --decisions file it was handed, and exits FAKE_EXIT.
func fake(t *testing.T, fixtures map[string]string) (Client, string) {
	t.Helper()
	dir := t.TempDir()
	for name, body := range fixtures {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	record := filepath.Join(dir, "argv")
	script := `#!/bin/sh
printf '%s\n' "$*" >> "` + record + `"
for a; do last=$a; done
case "$2" in
  status) cat "` + dir + `/status.json" ;;
  inspect) cat "` + dir + `/gate.json" ;;
  review) cp "$last" "` + record + `.decision" ;;
  reset) echo "Pending: review-artifact, apply-artifact" ;;
  run) cat "` + dir + `/events.jsonl"; sleep "${FAKE_SLEEP:-0}" ;;
esac
echo "${FAKE_STDERR:-}" >&2
exit "${FAKE_EXIT:-0}"
`
	binary := filepath.Join(dir, "eegtable")
	if err := os.WriteFile(binary, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	return Client{Binary: binary, Config: "study.yaml", Jobs: 1}, record
}

func argv(t *testing.T, record string) string {
	t.Helper()
	data, err := os.ReadFile(record)
	if err != nil {
		t.Fatal(err)
	}
	return strings.TrimSpace(string(data))
}

const statusFixture = `{"recordings": [
  {"label": "sub-01", "summary": "awaiting review-raw",
   "stages": [{"stage": "load", "state": "completed", "reason": ""},
              {"stage": "review-raw", "state": "needs-review", "reason": ""}],
   "next": {"kind": "review", "stage": "review-raw", "target": "raw"}},
  {"label": "sub-02", "summary": "exported", "stages": [], "next": null}]}`

func TestStatusParsesRecordingsAndNextAction(t *testing.T) {
	client, record := fake(t, map[string]string{"status.json": statusFixture})
	status, err := client.Status(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if got := argv(t, record); got != "preprocess status study.yaml --json" {
		t.Fatalf("argv = %q", got)
	}
	if len(status.Recordings) != 2 || status.Recordings[0].Label != "sub-01" {
		t.Fatalf("recordings = %+v", status.Recordings)
	}
	first := status.Recordings[0]
	if first.Next == nil || *first.Next != (Next{Kind: "review", Stage: "review-raw", Target: "raw"}) {
		t.Fatalf("next = %+v", first.Next)
	}
	if first.Stages[1].State != "needs-review" {
		t.Fatalf("stages = %+v", first.Stages)
	}
	if status.Recordings[1].Next != nil {
		t.Fatal("an exported recording has no next action")
	}
}

func TestFailureCarriesCommandAndStderr(t *testing.T) {
	client, _ := fake(t, map[string]string{"status.json": ""})
	t.Setenv("FAKE_EXIT", "2")
	t.Setenv("FAKE_STDERR", "eegtable preprocess: study.yaml: no such recipe")
	_, err := client.Status(context.Background())
	var failure *CommandError
	if !errors.As(err, &failure) {
		t.Fatalf("err = %v", err)
	}
	if failure.Code != 2 || !strings.Contains(failure.Stderr, "no such recipe") {
		t.Fatalf("failure = %+v", failure)
	}
	if !strings.Contains(err.Error(), "eegtable preprocess status study.yaml --json") {
		t.Fatalf("message = %q", err.Error())
	}
}

func TestMalformedJSONIsAnError(t *testing.T) {
	client, _ := fake(t, map[string]string{"status.json": "not json"})
	if _, err := client.Status(context.Background()); err == nil {
		t.Fatal("expected an error")
	}
}

const gateFixture = `{"stage": "review-raw", "parent_id": "abc", "field": "bads", "duration": 30.0,
  "items": [{"id": "Fp1", "label": "Fp1", "tags": ["eeg", "deviation"], "score": null, "suggested": true},
            {"id": "C3", "label": "C3", "tags": ["eeg"], "score": null, "suggested": false}],
  "spans": [{"onset": 5.0, "duration": 1.0, "description": "BAD_peak", "suggested": true}]}`

func TestInspectKeepsItemIdentitiesVerbatim(t *testing.T) {
	client, record := fake(t, map[string]string{"gate.json": gateFixture})
	gate, err := client.Inspect(context.Background(), "sub-01", "review-raw")
	if err != nil {
		t.Fatal(err)
	}
	want := "preprocess inspect study.yaml review-raw --recording sub-01 --json"
	if got := argv(t, record); got != want {
		t.Fatalf("argv = %q", got)
	}
	if gate.Field != "bads" || len(gate.Items) != 2 || string(gate.Items[0].ID) != `"Fp1"` {
		t.Fatalf("gate = %+v", gate)
	}
	if !gate.Items[0].Suggested || gate.Items[1].Suggested || gate.Spans[0].Description != "BAD_peak" {
		t.Fatalf("gate = %+v", gate)
	}
}

func TestReviewHandsPythonAJSONDecisionFile(t *testing.T) {
	client, record := fake(t, map[string]string{})
	decision := Decision{ParentID: "abc", Field: "bads", IDs: []json.RawMessage{json.RawMessage(`"Fp1"`)},
		Spans: []Span{{Onset: 5, Duration: 1, Description: "BAD_manual"}}}
	if err := client.Review(context.Background(), "sub-01", "raw", decision); err != nil {
		t.Fatal(err)
	}
	line := argv(t, record)
	if !strings.HasPrefix(line, "preprocess review study.yaml raw --recording sub-01 --decisions ") {
		t.Fatalf("argv = %q", line)
	}
	path := strings.TrimPrefix(line, "preprocess review study.yaml raw --recording sub-01 --decisions ")
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("temp decision %s should be removed", path)
	}
	copied, err := os.ReadFile(record + ".decision")
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	if err := json.Unmarshal(copied, &got); err != nil {
		t.Fatal(err)
	}
	spans := got["spans"].([]any)
	if got["parent_id"] != "abc" || got["bads"].([]any)[0] != "Fp1" || len(spans) != 1 {
		t.Fatalf("decision = %v", got)
	}
	if _, ok := got["fit_id"]; ok {
		t.Fatal("fit_id must be absent unless the gate provided one")
	}
}

func TestApplyDecisionIsABoolean(t *testing.T) {
	client, record := fake(t, map[string]string{})
	decision := Decision{ParentID: "abc", FitID: "fit", Field: "apply", IDs: []json.RawMessage{json.RawMessage(`0`)}}
	if err := client.Review(context.Background(), "sub-01", "artifact", decision); err != nil {
		t.Fatal(err)
	}
	copied, _ := os.ReadFile(record + ".decision")
	var got map[string]any
	if err := json.Unmarshal(copied, &got); err != nil {
		t.Fatal(err)
	}
	if got["apply"] != true || got["fit_id"] != "fit" {
		t.Fatalf("decision = %v", got)
	}
	if _, ok := got["spans"]; ok {
		t.Fatal("spans belong to the raw gate only")
	}
}

func TestResetReturnsWhatPythonRetired(t *testing.T) {
	client, record := fake(t, map[string]string{})
	out, err := client.Reset(context.Background(), "sub-01", "review-artifact")
	if err != nil {
		t.Fatal(err)
	}
	if got := argv(t, record); got != "preprocess reset study.yaml --from review-artifact --recording sub-01" {
		t.Fatalf("argv = %q", got)
	}
	if !strings.Contains(out, "review-artifact, apply-artifact") {
		t.Fatalf("out = %q", out)
	}
}

const eventsFixture = `{"event": "start", "subjects": ["sub-01"]}
{"event": "subject_start", "subject": "sub-01"}
{"event": "progress", "subject": "sub-01", "step": "load", "current": 1, "total": 9}
not json at all
{"event": "log", "level": "info", "subject": "sub-01", "message": "awaiting review-raw"}
`

func TestRunStreamsEventsAndTreatsExitThreeAsAGate(t *testing.T) {
	client, record := fake(t, map[string]string{"events.jsonl": eventsFixture})
	t.Setenv("FAKE_EXIT", "3")
	t.Setenv("FAKE_STDERR", "Filtering raw data")
	client.Jobs = 4
	process, err := client.Run("sub-01")
	if err != nil {
		t.Fatal(err)
	}
	var events []Event
	for event := range process.Events() {
		events = append(events, event)
	}
	result := <-process.Done()
	if got := argv(t, record); got != "preprocess run study.yaml --recording sub-01 --n-jobs 4 --progress-json" {
		t.Fatalf("argv = %q", got)
	}
	if len(events) != 6 || events[2].Step != "load" || events[4].Message != "awaiting review-raw" {
		t.Fatalf("events = %+v", events)
	}
	// Output that is not an event is surfaced rather than dropped: it is often
	// the only trace of what a worker process or a C extension was doing.
	if events[3] != (Event{Event: "stdout", Message: "not json at all"}) {
		t.Fatalf("stray stdout = %+v", events[3])
	}
	// stderr is streamed too, so a long ICA fit visibly isn't a hang.
	if events[5] != (Event{Event: "stderr", Message: "Filtering raw data"}) {
		t.Fatalf("stderr event = %+v", events[5])
	}
	if result.Code != 3 || result.Failed() || !strings.Contains(result.Stderr, "Filtering") {
		t.Fatalf("result = %+v", result)
	}
}

func TestRunAllOmitsTheRecordingFlag(t *testing.T) {
	client, record := fake(t, map[string]string{"events.jsonl": ""})
	process, err := client.Run("")
	if err != nil {
		t.Fatal(err)
	}
	for range process.Events() {
	}
	<-process.Done()
	if got := argv(t, record); got != "preprocess run study.yaml --n-jobs 1 --progress-json" {
		t.Fatalf("argv = %q", got)
	}
}

func TestRunForwardsNegativeAndZeroJobCounts(t *testing.T) {
	for _, jobs := range []int{-1, 0} {
		t.Run(strconv.Itoa(jobs), func(t *testing.T) {
			client, record := fake(t, map[string]string{"events.jsonl": ""})
			client.Jobs = jobs
			process, err := client.Run("")
			if err != nil {
				t.Fatal(err)
			}
			collect(t, process)
			want := "preprocess run study.yaml --n-jobs " + strconv.Itoa(jobs) + " --progress-json"
			if got := argv(t, record); got != want {
				t.Fatalf("argv = %q, want %q", got, want)
			}
		})
	}
}

func TestStopEndsARunningProcess(t *testing.T) {
	client, _ := fake(t, map[string]string{"events.jsonl": ""})
	t.Setenv("FAKE_SLEEP", "30")
	process, err := client.Run("sub-01")
	if err != nil {
		t.Fatal(err)
	}
	process.Stop()
	select {
	case result := <-process.Done():
		if !result.Failed() {
			t.Fatalf("a stopped run is not a success: %+v", result)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("process did not stop")
	}
}

func TestViewerCommandOpensTheCheckpoint(t *testing.T) {
	client, _ := fake(t, nil)
	cmd := client.Viewer("sub-01", "review-raw")
	want := []string{client.Binary, "preprocess", "inspect", "study.yaml", "review-raw", "--recording", "sub-01"}
	if strings.Join(cmd.Args, " ") != strings.Join(want, " ") {
		t.Fatalf("args = %v", cmd.Args)
	}
}

func TestLocateHonoursTheEnvironmentThenPath(t *testing.T) {
	t.Setenv("EEGTABLE", "/opt/venv/bin/eegtable")
	if got, err := Locate(); err != nil || got != "/opt/venv/bin/eegtable" {
		t.Fatalf("got %q, %v", got, err)
	}
	t.Setenv("EEGTABLE", "")
	t.Setenv("PATH", t.TempDir())
	if _, err := Locate(); err == nil || !strings.Contains(err.Error(), "eegtable[preprocessing]") {
		t.Fatalf("err = %v", err)
	}
}

func TestStopOnAProcessWithoutACommandIsHarmless(t *testing.T) {
	var _ Runner = (*Process)(nil)
	(&Process{}).Stop()
}

// collect drains a run, returning every event and the result.
func collect(t *testing.T, process Runner) []Event {
	t.Helper()
	var events []Event
	for event := range process.Events() {
		events = append(events, event)
	}
	<-process.Done()
	return events
}

func seen(events []Event, want string) bool {
	for _, event := range events {
		if strings.Contains(event.Message, want) {
			return true
		}
	}
	return false
}

func TestStrayStdoutReachesTheCallerInsteadOfVanishing(t *testing.T) {
	// Worker processes and C extensions write to the real stdout, which Python's
	// redirect never reaches. Dropping those lines hides real output.
	fixture := `{"event": "progress", "subject": "sub-01", "step": "load", "current": 1, "total": 7}
RANSAC interpolated 3 channels
{"event": "subject_done", "subject": "sub-01", "success": true}
`
	client, _ := fake(t, map[string]string{"events.jsonl": fixture})
	process, err := client.Run("sub-01")
	if err != nil {
		t.Fatal(err)
	}
	events := collect(t, process)
	if !seen(events, "RANSAC interpolated 3 channels") {
		t.Fatalf("a non-event stdout line must still reach the log: %+v", events)
	}
	if len(events) < 3 {
		t.Fatalf("the real events must survive alongside it: %+v", events)
	}
}

func TestProgressBarRedrawsArriveAsSeparateLines(t *testing.T) {
	// autoreject and pyprep redraw a bar with \r and no newline until it ends;
	// splitting only on \n makes the whole bar one line that never arrives.
	client, _ := fake(t, map[string]string{"events.jsonl": "Fitting:  10%\rFitting:  60%\rFitting: 100%\n"})
	process, err := client.Run("sub-01")
	if err != nil {
		t.Fatal(err)
	}
	events := collect(t, process)
	if !seen(events, "Fitting:  10%") || !seen(events, "Fitting: 100%") {
		t.Fatalf("each redraw is its own line: %+v", events)
	}
}

func TestAnEndlessLineIsCutInsteadOfEndingTheStream(t *testing.T) {
	// A line that never ends used to stop the reader in silence, which the TUI
	// reads as the run having finished.
	client, _ := fake(t, map[string]string{
		"events.jsonl": strings.Repeat("x", 3<<20) + "\n" + `{"event": "subject_done", "subject": "sub-01", "success": true}` + "\n"})
	process, err := client.Run("sub-01")
	if err != nil {
		t.Fatal(err)
	}
	events := collect(t, process)
	last := events[len(events)-1]
	if last.Event != "subject_done" {
		t.Fatalf("events after an endless line must still arrive, last = %+v", last)
	}
}

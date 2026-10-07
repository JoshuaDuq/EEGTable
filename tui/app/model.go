// Package app is the Bubble Tea front end: two screens over one Backend.
package app

import (
	"bytes"
	"context"
	"io"
	"os"
	"os/exec"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/JoshuaDuq/EEGTable/tui/eegtable"
	"github.com/JoshuaDuq/EEGTable/tui/styles"
)

type Backend interface {
	Status(ctx context.Context) (eegtable.Status, error)
	Inspect(ctx context.Context, recording, stage string) (eegtable.Gate, error)
	Review(ctx context.Context, recording, target string, decision eegtable.Decision) error
	Reset(ctx context.Context, recording, stage string) (string, error)
	Run(recording string) (eegtable.Runner, error)
	Viewer(recording, stage string) *exec.Cmd
}

type Model struct {
	backend Backend
	config  string
	refresh time.Duration
	width   int
	height  int

	status eegtable.Status
	// statusSeq numbers status loads so a slow one cannot land after a newer one;
	// statusInFlight keeps the idle poll to one Python process at a time.
	statusSeq      int
	statusInFlight bool
	statusErr      string

	cursor      int
	stagePane   bool
	stageCursor int
	busy        string
	notice      string
	failure     string
	confirm     *confirm
	run         *liveRun
	gate        *gateModel
	log         logView
	helpOpen    bool
	logOpen     bool
}

type confirm struct {
	question string
	detail   string
	verb     string
	yes      func(Model) (tea.Model, tea.Cmd)
}

const runDetail = "Finished checkpoints are kept; the stage in progress starts over on the next run."

type (
	statusMsg struct {
		seq    int
		status eegtable.Status
		err    error
	}
	gateMsg struct {
		recording string
		gate      eegtable.Gate
		err       error
	}
	savedMsg struct{ err error }
	resetMsg struct {
		out string
		err error
	}
	runStartedMsg struct {
		runner    eegtable.Runner
		recording string
		err       error
	}
	runEventMsg struct{ event eegtable.Event }
	runDoneMsg  struct{ result eegtable.Result }
	viewerMsg   struct{ failure string }
	tickMsg     struct{}
)

func New(backend Backend, config string) Model {
	return Model{backend: backend, config: config, refresh: 5 * time.Second, busy: "loading",
		statusSeq: 1, statusInFlight: true}
}

func (m Model) Init() tea.Cmd {
	return tea.Batch(m.loadStatus(), m.tick())
}

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width, m.height = msg.Width, msg.Height
		return m, nil
	case tickMsg:
		// The poll never sets busy: keys keep working while Python answers.
		if !m.statusInFlight && m.busy == "" && m.run == nil && m.gate == nil && m.confirm == nil {
			m.statusSeq++
			m.statusInFlight = true
			return m, tea.Batch(m.loadStatus(), m.tick())
		}
		return m, m.tick()
	case statusMsg:
		if msg.seq != m.statusSeq {
			return m, nil
		}
		m.statusInFlight = false
		if m.busy == "loading" {
			m.busy = ""
		}
		if msg.err != nil {
			m.statusErr = lastLine(msg.err.Error())
			return m, nil
		}
		m.statusErr = ""
		m.setStatus(msg.status)
		return m, nil
	case gateMsg:
		m.busy = ""
		if msg.err != nil {
			m.fail(msg.err.Error())
			return m, nil
		}
		gate := newGate(msg.recording, msg.gate)
		m.gate = &gate
		return m, nil
	case savedMsg:
		m.busy = ""
		if msg.err != nil {
			m.fail(msg.err.Error())
			return m, nil
		}
		m.notice = "Saved " + m.gate.recording + " " + m.gate.gate.Stage
		m.gate = nil
		return m.reload()
	case resetMsg:
		m.busy = ""
		if msg.err != nil {
			m.fail(msg.err.Error())
			return m, nil
		}
		m.notice = msg.out
		return m.reload()
	case runStartedMsg:
		m.busy = ""
		if msg.err != nil {
			m.fail(msg.err.Error())
			return m, nil
		}
		m.run = &liveRun{runner: msg.runner, recording: msg.recording, activeRecording: msg.recording, completed: map[string]map[string]bool{}}
		m.log.mark("run " + orAll(msg.recording))
		return m, m.wait()
	case runEventMsg:
		m.run.apply(msg.event)
		m.run.record(&m.log, msg.event)
		return m, m.wait()
	case runDoneMsg:
		// Only questions about the run can be open while it runs, and it has ended.
		m.confirm = nil
		switch {
		case msg.result.Failed():
			// stderr ends on whatever MNE narrated last, so a failure Python
			// reported is the cause; a signal (-1) explains itself in stderr.
			cause := msg.result.Stderr
			if m.run.failure != "" && msg.result.Code > 0 {
				cause = m.run.failure
			}
			m.fail("run exited " + itoa(msg.result.Code) + "\n" + cause)
		case msg.result.Code == 3:
			m.notice = "Run paused at a review gate"
			m.log.mark(m.notice)
		default:
			m.notice = "Run finished"
			m.log.mark(m.notice)
		}
		m.run = nil
		return m.reload()
	case viewerMsg:
		if msg.failure != "" {
			m.fail(msg.failure)
		}
		return m, nil
	case tea.KeyMsg:
		if msg.String() == "ctrl+c" {
			return m.quit()
		}
		if styles.IsTooSmall(m.width, m.height) {
			if msg.String() == "q" && (m.gate == nil || !m.gate.editing) {
				return m.requestQuit()
			}
			return m, nil
		}
		if m.confirm != nil {
			return m.updateConfirm(msg)
		}
		if m.helpOpen {
			switch msg.String() {
			case "?", "esc":
				m.helpOpen = false
			case "q":
				return m.requestQuit()
			}
			return m, nil
		}
		if msg.String() == "?" && (m.gate == nil || !m.gate.editing) {
			m.helpOpen = true
			return m, nil
		}
		if m.logOpen {
			return m.updateLog(msg)
		}
		if m.busy != "" {
			if msg.String() == "q" {
				return m.requestQuit()
			}
			return m, nil
		}
		if m.gate != nil {
			return m.updateGate(msg)
		}
		return m.updateHome(msg)
	}
	if m.gate != nil && m.gate.editing {
		var cmd tea.Cmd
		m.gate.input, cmd = m.gate.input.Update(msg)
		return m, cmd
	}
	return m, nil
}

func (m Model) View() string {
	if styles.IsTooSmall(m.width, m.height) {
		return styles.RenderTooSmall(m.width, m.height)
	}
	if m.confirm != nil {
		return m.viewConfirm()
	}
	if m.helpOpen {
		return m.viewHelp()
	}
	if m.logOpen {
		return m.viewLog()
	}
	if m.gate != nil {
		return m.viewGate()
	}
	return m.viewHome()
}

func (m Model) quit() (tea.Model, tea.Cmd) {
	if m.run != nil {
		m.run.runner.Stop()
	}
	return m, tea.Quit
}

// requestQuit asks first when quitting would throw work away; ctrl+c never asks.
func (m Model) requestQuit() (tea.Model, tea.Cmd) {
	switch {
	case m.run != nil:
		m.confirm = &confirm{"Stop the run and quit?", runDetail, "Stop and quit", Model.quit}
	case m.gate != nil && m.gate.edited():
		m.confirm = &confirm{"Quit without saving " + m.gate.gate.Stage + "?", "", "Quit", Model.quit}
	default:
		return m.quit()
	}
	return m, nil
}

func (m Model) updateConfirm(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "y":
		yes := m.confirm.yes
		m.confirm = nil
		return yes(m)
	case "n", "esc":
		m.confirm = nil
	}
	return m, nil
}

// fail keeps the whole text in the log and shows its last line, which is where
// Python puts the cause: its own one-line message, or a traceback's exception.
func (m *Model) fail(text string) {
	text = strings.TrimSpace(text)
	for _, line := range strings.Split(text, "\n") {
		m.log.add(styles.Fail.Render(line))
	}
	m.failure = lastLine(text)
}

func lastLine(text string) string {
	text = strings.TrimSpace(text)
	return text[strings.LastIndex(text, "\n")+1:]
}

// reload re-reads status after anything that changed the workspace; events and
// saves never decide state on their own.
func (m Model) reload() (tea.Model, tea.Cmd) {
	m.busy = "loading"
	m.statusSeq++
	m.statusInFlight = true
	return m, m.loadStatus()
}

func (m *Model) setStatus(status eegtable.Status) {
	selected := m.selectedLabel()
	stage := m.selectedStage().Stage
	m.status = status
	m.cursor = 0
	for i, recording := range status.Recordings {
		if recording.Label == selected {
			m.cursor = i
		}
	}
	m.stageCursor = 0
	if m.selectedLabel() == selected {
		for i, current := range m.visibleStages() {
			if current.Stage == stage {
				m.stageCursor = i
				break
			}
		}
	}
}

func (m Model) selectedLabel() string {
	if m.cursor < len(m.status.Recordings) {
		return m.status.Recordings[m.cursor].Label
	}
	return ""
}

func (m Model) loadStatus() tea.Cmd {
	backend, seq := m.backend, m.statusSeq
	return func() tea.Msg {
		status, err := backend.Status(context.Background())
		return statusMsg{seq, status, err}
	}
}

func (m Model) tick() tea.Cmd {
	return tea.Tick(m.refresh, func(time.Time) tea.Msg { return tickMsg{} })
}

func (m Model) openGate(recording, stage string) tea.Cmd {
	backend := m.backend
	return func() tea.Msg {
		gate, err := backend.Inspect(context.Background(), recording, stage)
		return gateMsg{recording, gate, err}
	}
}

func (m Model) save() tea.Cmd {
	backend, gate := m.backend, *m.gate
	return func() tea.Msg {
		target := trimPrefix(gate.gate.Stage, "review-")
		return savedMsg{backend.Review(context.Background(), gate.recording, target, gate.decision())}
	}
}

func (m Model) reset(recording, stage string) tea.Cmd {
	backend := m.backend
	return func() tea.Msg {
		out, err := backend.Reset(context.Background(), recording, stage)
		return resetMsg{out, err}
	}
}

func (m Model) startRun(recording string) tea.Cmd {
	backend := m.backend
	return func() tea.Msg {
		runner, err := backend.Run(recording)
		return runStartedMsg{runner, recording, err}
	}
}

// wait delivers the next run event, or the result once the events are drained.
func (m Model) wait() tea.Cmd {
	runner := m.run.runner
	return func() tea.Msg {
		if event, ok := <-runner.Events(); ok {
			return runEventMsg{event}
		}
		return runDoneMsg{<-runner.Done()}
	}
}

// viewer copies the viewer's stderr: Python's reason for failing is printed to
// the normal screen, which the TUI covers again as soon as the viewer exits.
func (m Model) viewer(recording, stage string) tea.Cmd {
	cmd := m.backend.Viewer(recording, stage)
	var stderr bytes.Buffer
	cmd.Stderr = io.MultiWriter(os.Stderr, &stderr)
	return tea.ExecProcess(cmd, func(err error) tea.Msg {
		if err == nil {
			return viewerMsg{}
		}
		return viewerMsg{"viewer: " + err.Error() + "\n" + stderr.String()}
	})
}

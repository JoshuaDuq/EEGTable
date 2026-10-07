package app

import (
	"context"
	"errors"
	"fmt"
	"os/exec"
	"slices"
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/JoshuaDuq/EEGTable/tui/eegtable"
	"github.com/JoshuaDuq/EEGTable/tui/styles"
)

type fakeRunner struct {
	events  chan eegtable.Event
	done    chan eegtable.Result
	stopped bool
}

func (r *fakeRunner) Events() <-chan eegtable.Event { return r.events }
func (r *fakeRunner) Done() <-chan eegtable.Result  { return r.done }
func (r *fakeRunner) Stop()                         { r.stopped = true }

type fakeBackend struct {
	status    eegtable.Status
	statusErr error
	gate      eegtable.Gate
	reviewErr error
	runner    *fakeRunner
	calls     []string
	decisions []eegtable.Decision
}

func (b *fakeBackend) Status(context.Context) (eegtable.Status, error) {
	b.calls = append(b.calls, "status")
	return b.status, b.statusErr
}

func (b *fakeBackend) Inspect(_ context.Context, recording, stage string) (eegtable.Gate, error) {
	b.calls = append(b.calls, "inspect "+recording+" "+stage)
	return b.gate, nil
}

func (b *fakeBackend) Review(_ context.Context, recording, target string, decision eegtable.Decision) error {
	b.calls = append(b.calls, "review "+recording+" "+target)
	b.decisions = append(b.decisions, decision)
	return b.reviewErr
}

func (b *fakeBackend) Reset(_ context.Context, recording, stage string) (string, error) {
	b.calls = append(b.calls, "reset "+recording+" "+stage)
	return "Pending: " + stage + ", export", nil
}

func (b *fakeBackend) Run(recording string) (eegtable.Runner, error) {
	b.calls = append(b.calls, "run "+recording)
	b.runner = &fakeRunner{events: make(chan eegtable.Event, 8), done: make(chan eegtable.Result, 1)}
	return b.runner, nil
}

func (b *fakeBackend) Viewer(recording, stage string) *exec.Cmd {
	b.calls = append(b.calls, "viewer "+recording+" "+stage)
	return exec.Command("true")
}

func (b *fakeBackend) called(call string) bool {
	for _, c := range b.calls {
		if c == call {
			return true
		}
	}
	return false
}

func (b *fakeBackend) count(call string) int {
	n := 0
	for _, c := range b.calls {
		if c == call {
			n++
		}
	}
	return n
}

func statusFixture() eegtable.Status {
	stage := func(name, state string) eegtable.Stage { return eegtable.Stage{Stage: name, State: state} }
	return eegtable.Status{Recordings: []eegtable.Recording{
		{Label: "sub-01", Summary: "awaiting review-raw",
			Stages: []eegtable.Stage{stage("load", "completed"), stage("crop-raw", "disabled"),
				stage("detect-bads", "completed"), stage("review-raw", "needs-review"),
				stage("filter", "pending"), stage("export", "pending")},
			Next: &eegtable.Next{Kind: "review", Stage: "review-raw", Target: "raw"}},
		{Label: "sub-02", Summary: "exported", Stages: []eegtable.Stage{stage("load", "completed")}},
		{Label: "sub-03", Summary: "stale at events", Stages: []eegtable.Stage{stage("events", "stale")},
			Next: &eegtable.Next{Kind: "reset", Stage: "events"}},
		{Label: "sub-04", Summary: "0 of 9 stages", Stages: []eegtable.Stage{stage("load", "pending")},
			Next: &eegtable.Next{Kind: "run", Stage: "load"}},
	}}
}

func score(v float64) *float64 { return &v }

func rawGate() eegtable.Gate {
	return eegtable.Gate{Stage: "review-raw", Parent: "detect-bads", ParentID: "abc", Field: "bads", Duration: 30,
		Items: []eegtable.Item{
			{ID: []byte(`"Fp1"`), Label: "Fp1", Tags: []string{"eeg", "deviation"}, Suggested: true},
			{ID: []byte(`"C3"`), Label: "C3", Tags: []string{"eeg"}},
			{ID: []byte(`"VEOG"`), Label: "VEOG", Tags: []string{"eog"}},
		},
		Spans: []eegtable.Span{{Onset: 5, Duration: 1, Description: "BAD_peak", Suggested: true}}}
}

func icaGate() eegtable.Gate {
	return eegtable.Gate{Stage: "review-artifact", Parent: "fit-artifact", ParentID: "abc", FitID: "fit",
		Method: "ica", Field: "exclude",
		Items: []eegtable.Item{
			{ID: []byte(`0`), Label: "ICA000", Tags: []string{"eye blink", "VEOG"}, Score: score(0.98), Suggested: true},
			{ID: []byte(`1`), Label: "ICA001", Tags: []string{"brain"}, Score: score(0.2)},
			{ID: []byte(`2`), Label: "ICA002", Tags: []string{"muscle"}, Score: score(0.91), Suggested: true},
			{ID: []byte(`3`), Label: "ICA003", Tags: []string{"other"}},
		}}
}

func update(t *testing.T, m Model, msg tea.Msg) (Model, tea.Cmd) {
	t.Helper()
	next, cmd := m.Update(msg)
	return next.(Model), cmd
}

// messages runs one command and returns what it produced, unpacking a batch.
func messages(cmd tea.Cmd) []tea.Msg {
	if cmd == nil {
		return nil
	}
	msg := cmd()
	// Tests advance the periodic timer explicitly with tickMsg.
	if _, ok := msg.(tickMsg); ok {
		return nil
	}
	batch, ok := msg.(tea.BatchMsg)
	if !ok {
		return []tea.Msg{msg}
	}
	var out []tea.Msg
	for _, c := range batch {
		if c != nil {
			out = append(out, messages(c)...)
		}
	}
	return out
}

// settle feeds every message a command produces back into the model, once.
func settle(t *testing.T, m Model, cmd tea.Cmd) Model {
	t.Helper()
	m, _ = step(t, m, cmd)
	return m
}

// step is settle that also hands back the command the last message produced.
func step(t *testing.T, m Model, cmd tea.Cmd) (Model, tea.Cmd) {
	t.Helper()
	var next tea.Cmd
	for _, msg := range messages(cmd) {
		m, next = update(t, m, msg)
	}
	return m, next
}

func key(r rune) tea.KeyMsg            { return tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{r}} }
func special(k tea.KeyType) tea.KeyMsg { return tea.KeyMsg{Type: k} }

func home(t *testing.T, backend *fakeBackend) Model {
	t.Helper()
	m := New(backend, "study.yaml")
	m.refresh = time.Millisecond
	m, _ = update(t, m, tea.WindowSizeMsg{Width: 120, Height: 40})
	return settle(t, m, m.Init())
}

func openGate(t *testing.T, backend *fakeBackend) Model {
	t.Helper()
	m := home(t, backend)
	m, cmd := update(t, m, special(tea.KeyEnter))
	return settle(t, m, cmd)
}

func TestInitLoadsStatusAndRendersRecordings(t *testing.T) {
	backend := &fakeBackend{status: statusFixture()}
	view := home(t, backend).View()
	if backend.calls[0] != "status" {
		t.Fatalf("calls = %v", backend.calls)
	}
	for _, want := range []string{"RECORDINGS", "sub-01", "awaiting review-raw", "sub-02", "exported",
		"detect-bads", "needs review", "review raw"} {
		if !strings.Contains(view, want) {
			t.Fatalf("view lacks %q:\n%s", want, view)
		}
	}
	if strings.Contains(view, "crop-raw") {
		t.Fatal("disabled stages are noise")
	}
}

func TestStatusFailureIsShownVerbatim(t *testing.T) {
	backend := &fakeBackend{statusErr: errors.New("eegtable preprocess status study.yaml --json: exit 2\nstudy.yaml: no such recipe")}
	if view := home(t, backend).View(); !strings.Contains(view, "no such recipe") {
		t.Fatalf("view = %s", view)
	}
}

func TestEnterOpensTheGateWithSuggestionsTicked(t *testing.T) {
	backend := &fakeBackend{status: statusFixture(), gate: rawGate()}
	view := openGate(t, backend).View()
	if !backend.called("inspect sub-01 review-raw") {
		t.Fatalf("calls = %v", backend.calls)
	}
	for _, want := range []string{"CHANNELS", "▣ Fp1", "□ C3", "deviation", "SPANS", "5.00", "BAD_peak", "onset duration", "1 marked bad"} {
		if !strings.Contains(view, want) {
			t.Fatalf("view lacks %q:\n%s", want, view)
		}
	}
}

func TestGateSaveSendsTheTickedRowsAndKeptSpans(t *testing.T) {
	backend := &fakeBackend{status: statusFixture(), gate: rawGate()}
	m := openGate(t, backend)
	m, _ = update(t, m, key('a'))
	m, _ = update(t, m, special(tea.KeyDown))
	m, _ = update(t, m, key(' '))
	m, cmd := update(t, m, special(tea.KeyEnter))
	m, next := step(t, m, cmd)
	m = settle(t, m, next)
	if !backend.called("review sub-01 raw") || len(backend.decisions) != 1 {
		t.Fatalf("calls = %v", backend.calls)
	}
	decision := backend.decisions[0]
	if decision.ParentID != "abc" || decision.Field != "bads" || len(decision.IDs) != 1 || string(decision.IDs[0]) != `"C3"` {
		t.Fatalf("decision = %+v", decision)
	}
	if len(decision.Spans) != 1 || decision.Spans[0].Description != "BAD_peak" {
		t.Fatalf("spans = %+v", decision.Spans)
	}
	view := m.View()
	if !strings.Contains(view, "RECORDINGS") || !strings.Contains(view, "Saved") || backend.count("status") != 2 {
		t.Fatalf("after save: calls=%v\n%s", backend.calls, view)
	}
}

func TestRestoringSuggestionsAfterClearing(t *testing.T) {
	backend := &fakeBackend{status: statusFixture(), gate: rawGate()}
	m := openGate(t, backend)
	m, _ = update(t, m, key('a'))
	if view := m.View(); strings.Contains(view, "▣ Fp1") || !strings.Contains(view, "BAD_peak") {
		t.Fatalf("clear unticks channels and leaves spans alone:\n%s", view)
	}
	m, _ = update(t, m, key('s'))
	if view := m.View(); !strings.Contains(view, "▣ Fp1") || !strings.Contains(view, "□ C3") {
		t.Fatalf("view = %s", view)
	}
}

func TestSaveFailureStaysOnTheGateWithPythonsMessage(t *testing.T) {
	backend := &fakeBackend{status: statusFixture(), gate: rawGate(),
		reviewErr: &eegtable.CommandError{Code: 2, Stderr: "review.parent_id: stale or missing reviewed checkpoint identity"}}
	m := openGate(t, backend)
	m, cmd := update(t, m, special(tea.KeyEnter))
	m = settle(t, m, cmd)
	view := m.View()
	if !strings.Contains(view, "CHANNELS") || !strings.Contains(view, "stale or missing") {
		t.Fatalf("view = %s", view)
	}
}

func TestSpanInputValidatesBeforeAppending(t *testing.T) {
	backend := &fakeBackend{status: statusFixture(), gate: rawGate()}
	m := openGate(t, backend)
	for i := 0; i < 4; i++ {
		m, _ = update(t, m, special(tea.KeyDown))
	}
	m, _ = update(t, m, special(tea.KeyEnter))
	m, _ = update(t, m, tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("abc")})
	m, _ = update(t, m, special(tea.KeyEnter))
	if view := m.View(); !strings.Contains(view, "expected") {
		t.Fatalf("invalid input must be refused:\n%s", view)
	}
	m, _ = update(t, m, special(tea.KeyCtrlU))
	m, _ = update(t, m, tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("29 5")})
	m, _ = update(t, m, special(tea.KeyEnter))
	if !strings.Contains(m.View(), "30.0") {
		t.Fatalf("a span past the recording end must name the limit:\n%s", m.View())
	}
	m, _ = update(t, m, special(tea.KeyCtrlU))
	m, _ = update(t, m, tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("12.5 2 BAD_manual")})
	m, _ = update(t, m, special(tea.KeyEnter))
	if view := m.View(); !strings.Contains(view, "12.50") || !strings.Contains(view, "BAD_manual") {
		t.Fatalf("view = %s", view)
	}
	m, cmd := update(t, m, special(tea.KeyEnter))
	settle(t, m, cmd)
	spans := backend.decisions[0].Spans
	if len(spans) != 2 || spans[1] != (eegtable.Span{Onset: 12.5, Duration: 2, Description: "BAD_manual"}) {
		t.Fatalf("spans = %+v", spans)
	}
}

func TestSpanWithoutADescriptionIsBADManual(t *testing.T) {
	backend := &fakeBackend{status: statusFixture(), gate: rawGate()}
	m := openGate(t, backend)
	for i := 0; i < 4; i++ {
		m, _ = update(t, m, special(tea.KeyDown))
	}
	m, _ = update(t, m, special(tea.KeyEnter))
	m, _ = update(t, m, tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("1 0.5")})
	m, _ = update(t, m, special(tea.KeyEnter))
	if !strings.Contains(m.View(), "BAD_manual") {
		t.Fatalf("view = %s", m.View())
	}
}

func TestSortByScoreOrdersComponents(t *testing.T) {
	backend := &fakeBackend{status: statusFixture(), gate: icaGate()}
	m := openGate(t, backend)
	view := m.View()
	if !strings.Contains(view, "COMPONENTS") || !strings.Contains(view, "0.98") || !strings.Contains(view, "eye blink") {
		t.Fatalf("view = %s", view)
	}
	m, _ = update(t, m, key('o'))
	view = m.View()
	order := []int{strings.Index(view, "ICA000"), strings.Index(view, "ICA002"), strings.Index(view, "ICA001"), strings.Index(view, "ICA003")}
	for i := 1; i < len(order); i++ {
		if order[i-1] < 0 || order[i-1] > order[i] {
			t.Fatalf("order = %v\n%s", order, view)
		}
	}
	m, cmd := update(t, m, special(tea.KeyEnter))
	settle(t, m, cmd)
	decision := backend.decisions[0]
	ids := fmt.Sprintf("%s %s", decision.IDs[0], decision.IDs[1])
	if decision.FitID != "fit" || decision.Field != "exclude" || len(decision.IDs) != 2 || ids != "0 2" {
		t.Fatalf("decision = %+v", decision)
	}
}

func TestRunStreamsIntoTheStagePane(t *testing.T) {
	backend := &fakeBackend{status: statusFixture()}
	m := home(t, backend)
	m, cmd := update(t, m, key('r'))
	m = settle(t, m, cmd)
	if !backend.called("run sub-01") || !strings.Contains(m.View(), "running") {
		t.Fatalf("calls=%v\n%s", backend.calls, m.View())
	}
	backend.runner.events <- eegtable.Event{Event: "progress", Subject: "sub-01", Step: "filter", Current: 5, Total: 9}
	backend.runner.events <- eegtable.Event{Event: "stderr", Message: "Filtering raw data"}
	m = settle(t, m, m.wait())
	m = settle(t, m, m.wait())
	view := m.View()
	if !strings.Contains(view, "filter") || !strings.Contains(view, "Filtering raw data") {
		t.Fatalf("view = %s", view)
	}
	close(backend.runner.events)
	backend.runner.done <- eegtable.Result{Code: 3}
	m, next := step(t, m, m.wait())
	m = settle(t, m, next)
	if backend.count("status") != 2 || strings.Contains(m.View(), "running") || !strings.Contains(m.View(), "paused at a review gate") {
		t.Fatalf("a finished run refreshes status: calls=%v\n%s", backend.calls, m.View())
	}
}

// A recipe that stops loading before any recording starts reports on stderr alone.
func TestFailedRunWithoutAnErrorEventShowsTheStderrTail(t *testing.T) {
	backend := &fakeBackend{status: statusFixture()}
	m := home(t, backend)
	m, cmd := update(t, m, key('r'))
	m = settle(t, m, cmd)
	close(backend.runner.events)
	backend.runner.done <- eegtable.Result{Code: 2, Stderr: "eegtable preprocess: inputs.root: no such directory"}
	m = settle(t, m, m.wait())
	if !strings.Contains(m.View(), "no such directory") {
		t.Fatalf("view = %s", m.View())
	}
}

// The real --progress-json failure: Python sends the cause as an error-level log
// event on stdout, and stderr ends on whatever MNE narrated last.
func TestFailedRunShowsPythonsErrorEventOverMNENarration(t *testing.T) {
	previous := lipgloss.ColorProfile()
	lipgloss.SetColorProfile(0) // termenv.TrueColor, so the failure style shows in the text
	t.Cleanup(func() { lipgloss.SetColorProfile(previous) })
	narration := "Reading 0 ... 4999  =      0.000 ...    19.996 secs..."
	cause := "Could not find any of the events you specified."
	run := func(t *testing.T, result eegtable.Result) Model {
		backend := &fakeBackend{status: statusFixture()}
		m := startRun(t, backend, 40)
		m = feed(t, m, backend,
			eegtable.Event{Event: "stderr", Message: narration},
			eegtable.Event{Event: "log", Level: "error", Subject: "sub-01", Message: cause},
			eegtable.Event{Event: "subject_done", Subject: "sub-01"},
			eegtable.Event{Event: "log", Level: "warning", Message: "✗ 1 of 1 recordings failed: sub-01"},
			eegtable.Event{Event: "complete"},
			eegtable.Event{Event: "stderr", Message: narration})
		close(backend.runner.events)
		backend.runner.done <- result
		return settle(t, m, m.wait())
	}
	m := run(t, eegtable.Result{Code: 1, Stderr: narration})
	if m.failure != cause || !strings.Contains(m.View(), cause) {
		t.Fatalf("failure = %q", m.failure)
	}
	if !slices.Contains(m.log.lines, styles.Fail.Render(cause)) {
		t.Fatalf("the cause must be styled as a failure in the log:\n%q", m.log.lines)
	}
	if slices.Contains(m.log.lines, styles.Fail.Render(narration)) {
		t.Fatalf("MNE's narration is not a failure:\n%q", m.log.lines)
	}
	// A signal ended the run, not the recording that failed before it.
	m = run(t, eegtable.Result{Code: -1, Stderr: narration + "\nsignal: terminated"})
	if m.failure != "signal: terminated" {
		t.Fatalf("failure = %q", m.failure)
	}
}

func TestStoppingALiveRunAsksFirst(t *testing.T) {
	backend := &fakeBackend{status: statusFixture()}
	m := startRun(t, backend, 40)
	for _, press := range []tea.KeyMsg{special(tea.KeyEsc), key('q')} {
		m, _ = update(t, m, press)
		if backend.runner.stopped || m.confirm == nil || !strings.Contains(m.View(), "Stop the run") {
			t.Fatalf("%s stopped the run without asking", press.String())
		}
		m, _ = update(t, m, key('n'))
		if backend.runner.stopped || m.confirm != nil || m.run == nil {
			t.Fatal("cancel must leave the run alone")
		}
	}
	m, _ = update(t, m, special(tea.KeyEsc))
	m, _ = update(t, m, key('y'))
	if !backend.runner.stopped || !strings.Contains(m.View(), "Stopping") {
		t.Fatal("confirmed stop must stop the run")
	}
}

func TestQuittingFromTheLogAsksWhileARunIsLive(t *testing.T) {
	backend := &fakeBackend{status: statusFixture()}
	m := startRun(t, backend, 40)
	m, _ = update(t, m, key('l'))
	m, cmd := update(t, m, key('q'))
	if backend.runner.stopped || m.confirm == nil || cmd != nil {
		t.Fatal("q in the log stopped the run without asking")
	}
	m, _ = update(t, m, key('n'))
	if backend.runner.stopped || m.run == nil || !m.logOpen {
		t.Fatal("cancel must leave the run and the log alone")
	}
}

func TestRunEndingWithdrawsItsQuestion(t *testing.T) {
	backend := &fakeBackend{status: statusFixture()}
	m := startRun(t, backend, 40)
	m, _ = update(t, m, special(tea.KeyEsc))
	m, cmd := update(t, m, runDoneMsg{eegtable.Result{Code: 0}})
	m = settle(t, m, cmd)
	if m.confirm != nil {
		t.Fatal("a finished run still asks whether to stop it")
	}
	if _, cmd = update(t, m, key('y')); cmd != nil {
		t.Fatal("a stale answer dispatched an action")
	}
}

func TestLeavingAnEditedGateAsksFirst(t *testing.T) {
	backend := &fakeBackend{status: statusFixture(), gate: rawGate()}
	m := openGate(t, backend)
	m, cmd := update(t, m, special(tea.KeyEsc))
	if m.gate != nil || cmd == nil {
		t.Fatal("an untouched gate closes without asking")
	}
	m = openGate(t, backend)
	m, _ = update(t, m, key(' '))
	for _, press := range []tea.KeyMsg{special(tea.KeyEsc), key('q')} {
		m, cmd = update(t, m, press)
		if cmd != nil || m.confirm == nil {
			t.Fatalf("%s discarded an edited decision without asking", press.String())
		}
		m, _ = update(t, m, special(tea.KeyEsc))
		if m.gate == nil || !m.gate.edited() {
			t.Fatal("cancel lost the edits")
		}
	}
	m, _ = update(t, m, special(tea.KeyEsc))
	m, cmd = update(t, m, key('y'))
	if m.gate != nil || cmd == nil || len(backend.decisions) != 0 {
		t.Fatal("confirmed discard must close the gate unsaved")
	}
}

func TestFailuresShowPythonsReasonAndLogEverything(t *testing.T) {
	backend := &fakeBackend{status: statusFixture(), gate: rawGate(),
		reviewErr: &eegtable.CommandError{Args: []string{"preprocess", "review", "/very/long/path/study.yaml", "raw"},
			Code: 2, Stderr: "eegtable preprocess: review.parent_id: stale checkpoint identity"}}
	m := openGate(t, backend)
	m, cmd := update(t, m, special(tea.KeyEnter))
	m = settle(t, m, cmd)
	if m.failure != "eegtable preprocess: review.parent_id: stale checkpoint identity" {
		t.Fatalf("failure = %q", m.failure)
	}
	if log := strings.Join(m.log.lines, "\n"); !strings.Contains(log, "/very/long/path/study.yaml") {
		t.Fatalf("log lacks the command:\n%s", log)
	}
}

func TestRunAllAndNextActionsFromHome(t *testing.T) {
	backend := &fakeBackend{status: statusFixture()}
	m := home(t, backend)
	m, cmd := update(t, m, key('a'))
	settle(t, m, cmd)
	if !backend.called("run ") {
		t.Fatalf("calls = %v", backend.calls)
	}
	backend = &fakeBackend{status: statusFixture()}
	m = home(t, backend)
	for i := 0; i < 3; i++ {
		m, _ = update(t, m, special(tea.KeyDown))
	}
	m, cmd = update(t, m, special(tea.KeyEnter))
	settle(t, m, cmd)
	if !backend.called("run sub-04") {
		t.Fatalf("enter on a pending recording runs it: %v", backend.calls)
	}
	m = home(t, backend)
	m, _ = update(t, m, special(tea.KeyEnd))
	m, _ = update(t, m, special(tea.KeyUp))
	m, _ = update(t, m, special(tea.KeyEnter))
	if !strings.Contains(m.View(), "Reset sub-03 from events?") {
		t.Fatalf("enter on a stale recording asks before resetting:\n%s", m.View())
	}
}

func TestResetNeedsConfirmation(t *testing.T) {
	backend := &fakeBackend{status: statusFixture()}
	m := home(t, backend)
	m, _ = update(t, m, special(tea.KeyTab))
	m, _ = update(t, m, special(tea.KeyDown))
	m, _ = update(t, m, key('x'))
	if !strings.Contains(m.View(), "Reset sub-01 from detect-bads?") {
		t.Fatalf("view = %s", m.View())
	}
	m, _ = update(t, m, key('n'))
	if strings.Contains(m.View(), "Reset sub-01") || backend.called("reset sub-01 detect-bads") {
		t.Fatal("n cancels")
	}
	m, _ = update(t, m, key('x'))
	m, cmd := update(t, m, key('y'))
	m, next := step(t, m, cmd)
	m = settle(t, m, next)
	if !backend.called("reset sub-01 detect-bads") || !strings.Contains(m.View(), "Pending: detect-bads") || backend.count("status") != 2 {
		t.Fatalf("calls=%v\n%s", backend.calls, m.View())
	}
}

func TestViewerOpensTheReviewedCheckpoint(t *testing.T) {
	backend := &fakeBackend{status: statusFixture(), gate: rawGate()}
	m := openGate(t, backend)
	m, _ = update(t, m, key('v'))
	if !backend.called("viewer sub-01 detect-bads") {
		t.Fatalf("calls = %v", backend.calls)
	}
	m, cmd := update(t, m, special(tea.KeyEsc))
	m = settle(t, m, cmd)
	m, _ = update(t, m, special(tea.KeyTab))
	m, _ = update(t, m, key('v'))
	if !backend.called("viewer sub-01 load") {
		t.Fatalf("home viewer opens the selected completed stage: %v", backend.calls)
	}
	m, _ = update(t, m, special(tea.KeyDown))
	m, _ = update(t, m, special(tea.KeyDown))
	m, _ = update(t, m, key('v'))
	if backend.count("viewer sub-01 review-raw") != 0 {
		t.Fatal("a stage without a checkpoint has nothing to open")
	}
}

func TestIdleTickRefreshesUnlessBusy(t *testing.T) {
	backend := &fakeBackend{status: statusFixture()}
	m := home(t, backend)
	m, cmd := update(t, m, tickMsg{})
	m = settle(t, m, cmd)
	if backend.count("status") != 2 {
		t.Fatalf("idle tick refreshes: %v", backend.calls)
	}
	m, cmd = update(t, m, key('r'))
	m = settle(t, m, cmd)
	_, cmd = update(t, m, tickMsg{})
	messages(cmd)
	if backend.count("status") != 2 {
		t.Fatalf("a running model does not poll: %v", backend.calls)
	}
}

func TestSmallTerminalAndQuit(t *testing.T) {
	backend := &fakeBackend{status: statusFixture()}
	m := home(t, backend)
	m, _ = update(t, m, tea.WindowSizeMsg{Width: 50, Height: 10})
	if !strings.Contains(m.View(), "too small") {
		t.Fatalf("view = %s", m.View())
	}
	_, cmd := update(t, m, key('q'))
	if _, ok := messages(cmd)[0].(tea.QuitMsg); !ok {
		t.Fatal("q quits")
	}
}

func TestLongLabelsKeepTheirSummaryVisible(t *testing.T) {
	status := eegtable.Status{Recordings: []eegtable.Recording{{
		Label: "sub-0007_task-thermalactive_run-1", Summary: "0 of 22 stages",
		Stages: []eegtable.Stage{{Stage: "load", State: "pending"}},
		Next:   &eegtable.Next{Kind: "run", Stage: "load"}}}}
	backend := &fakeBackend{status: status}
	view := home(t, backend).View()
	if !strings.Contains(view, "0 of 22 stages") {
		t.Fatalf("summary truncated:\n%s", view)
	}
	if strings.Contains(view, "1 recordings") || !strings.Contains(view, "1 recording") {
		t.Fatalf("plural:\n%s", view)
	}
}

func TestHeaderShowsTheRecipeName(t *testing.T) {
	backend := &fakeBackend{status: statusFixture()}
	m := New(backend, "/Volumes/KINGSTON/EEG_fMRI_data/EEGFeat/sub-0007/eeg/sub-0007_task-thermalactive_run-1_recipe.yaml")
	m.refresh = time.Millisecond
	m, _ = update(t, m, tea.WindowSizeMsg{Width: 100, Height: 30})
	m = settle(t, m, m.Init())
	first := strings.SplitN(m.View(), "\n", 2)[0]
	if !strings.Contains(first, "run-1_recipe.yaml") || !strings.Contains(first, "4 recordings") {
		t.Fatalf("header must keep the recipe name and the count on one line: %q", first)
	}
}

func startRun(t *testing.T, backend *fakeBackend, height int) Model {
	t.Helper()
	m := New(backend, "study.yaml")
	m.refresh = time.Millisecond
	m, _ = update(t, m, tea.WindowSizeMsg{Width: 120, Height: height})
	m = settle(t, m, m.Init())
	m, cmd := update(t, m, key('r'))
	return settle(t, m, cmd)
}

func feed(t *testing.T, m Model, backend *fakeBackend, events ...eegtable.Event) Model {
	t.Helper()
	for _, event := range events {
		backend.runner.events <- event
		m = settle(t, m, m.wait())
	}
	return m
}

func TestRunOutputAccumulatesInTheLogPanelAndOutlivesTheRun(t *testing.T) {
	backend := &fakeBackend{status: statusFixture()}
	m := startRun(t, backend, 50)
	m = feed(t, m, backend,
		eegtable.Event{Event: "stderr", Message: "Filtering raw data"},
		eegtable.Event{Event: "log", Level: "info", Subject: "sub-01", Message: "awaiting review-raw"})
	view := m.View()
	for _, want := range []string{"LOG", "run sub-01", "Filtering raw data", "awaiting review-raw"} {
		if !strings.Contains(view, want) {
			t.Fatalf("log lacks %q:\n%s", want, view)
		}
	}
	close(backend.runner.events)
	backend.runner.done <- eegtable.Result{Code: 3}
	m, next := step(t, m, m.wait())
	m = settle(t, m, next)
	view = m.View()
	if !strings.Contains(view, "Filtering raw data") || !strings.Contains(view, "paused at a review gate") {
		t.Fatalf("log must outlive the run:\n%s", view)
	}
}

func TestLogPanelYieldsToALongStageList(t *testing.T) {
	status := statusFixture()
	for i := 0; i < 20; i++ {
		status.Recordings[0].Stages = append(status.Recordings[0].Stages, eegtable.Stage{Stage: fmt.Sprintf("stage-%02d", i), State: "pending"})
	}
	backend := &fakeBackend{status: status}
	m := startRun(t, backend, 24)
	m = feed(t, m, backend, eegtable.Event{Event: "stderr", Message: "Filtering raw data"})
	if view := m.View(); strings.Contains(view, "LOG") || !strings.Contains(view, "load") || strings.Contains(view, "stage-19") {
		t.Fatalf("no room for a log at 24 rows with 25 stages:\n%s", view)
	}
	m, _ = update(t, m, special(tea.KeyTab))
	for i := 0; i < 24; i++ {
		m, _ = update(t, m, special(tea.KeyDown))
	}
	if view := m.View(); !strings.Contains(view, "stage-19") || !strings.Contains(view, "more") {
		t.Fatalf("the stage list follows its cursor:\n%s", view)
	}
}

func TestLogScrollsWithPageKeys(t *testing.T) {
	backend := &fakeBackend{status: statusFixture()}
	m := startRun(t, backend, 40)
	for i := 0; i < 60; i++ {
		m = feed(t, m, backend, eegtable.Event{Event: "stderr", Message: fmt.Sprintf("line %02d", i)})
	}
	if view := m.View(); !strings.Contains(view, "line 59") || strings.Contains(view, "line 00") {
		t.Fatalf("newest lines show by default:\n%s", view)
	}
	m, _ = update(t, m, special(tea.KeyPgUp))
	if view := m.View(); strings.Contains(view, "line 59") {
		t.Fatalf("page up scrolls back:\n%s", view)
	}
	m, _ = update(t, m, special(tea.KeyPgDown))
	if view := m.View(); !strings.Contains(view, "line 59") {
		t.Fatalf("page down returns to the newest lines:\n%s", view)
	}
}

// logText is every line the log holds, whatever the panel has room for.
func logText(m Model) string { return m.log.view(len(m.log.lines)) }

func TestLogTrailsEachStageAsItFinishes(t *testing.T) {
	backend := &fakeBackend{status: statusFixture()}
	m := startRun(t, backend, 50)
	m = feed(t, m, backend,
		eegtable.Event{Event: "progress", Subject: "sub-01", Step: "load", Current: 1, Total: 7},
		eegtable.Event{Event: "progress", Subject: "sub-01", Step: "detect-bads", Current: 6, Total: 7},
		eegtable.Event{Event: "progress", Subject: "sub-01", Step: "review-raw", Current: 7, Total: 7})
	log := logText(m)
	for _, want := range []string{"1/7", "load", "6/7", "detect-bads", "7/7", "review-raw"} {
		if !strings.Contains(log, want) {
			t.Fatalf("the log must name each finished stage, lacks %q:\n%s", want, log)
		}
	}
	if strings.Contains(log, "✓ review-raw") {
		t.Fatalf("a review gate has not completed, it is waiting:\n%s", log)
	}
}

func TestLogNamesTheRecordingWhenRunningAll(t *testing.T) {
	backend := &fakeBackend{status: statusFixture()}
	m := New(backend, "study.yaml")
	m.refresh = time.Millisecond
	m, _ = update(t, m, tea.WindowSizeMsg{Width: 120, Height: 50})
	m = settle(t, m, m.Init())
	m, cmd := update(t, m, key('a'))
	m = settle(t, m, cmd)
	m = feed(t, m, backend,
		eegtable.Event{Event: "progress", Subject: "sub-01", Step: "load", Current: 1, Total: 2},
		eegtable.Event{Event: "subject_start", Subject: "sub-02"},
		eegtable.Event{Event: "progress", Subject: "sub-02", Step: "load", Current: 1, Total: 2})
	if log := logText(m); !strings.Contains(log, "sub-01") || !strings.Contains(log, "sub-02") {
		t.Fatalf("a run over every recording must say which one each line is about:\n%s", log)
	}
}

func TestErrorEventsReachTheLog(t *testing.T) {
	backend := &fakeBackend{status: statusFixture()}
	m := startRun(t, backend, 50)
	m = feed(t, m, backend,
		eegtable.Event{Event: "error", Message: "inputs.root: no such directory"})
	if log := logText(m); !strings.Contains(log, "no such directory") {
		t.Fatalf("a run that fails before processing must say so:\n%s", log)
	}
}

func TestMultiLineMessagesBecomeSeparateLogLines(t *testing.T) {
	backend := &fakeBackend{status: statusFixture()}
	m := startRun(t, backend, 50)
	before := len(m.log.lines)
	m = feed(t, m, backend, eegtable.Event{Event: "log", Level: "info",
		Message: "1 of 1 recordings await review\nNext: eegtable preprocess review recipe.yaml"})
	if got := len(m.log.lines) - before; got != 2 {
		t.Fatalf("a two-line message is two log lines, not %d: %q", got, m.log.lines[before:])
	}
}

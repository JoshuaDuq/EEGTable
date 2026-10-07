package app

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/JoshuaDuq/EEGTable/tui/eegtable"
	"github.com/JoshuaDuq/EEGTable/tui/styles"
)

// liveRun mirrors a running `eegtable preprocess run` for display only: which
// stages each recording has finished and the last lines Python wrote.
type liveRun struct {
	runner          eegtable.Runner
	recording       string
	activeRecording string
	completed       map[string]map[string]bool
	// failure is the last recording failure Python reported. It arrives as an
	// error-level log event, while stderr carries MNE's narration, not the cause.
	failure string
}

func (r *liveRun) apply(event eegtable.Event) {
	switch event.Event {
	case "subject_start":
		r.activeRecording = event.Subject
	case "subject_done":
		if r.activeRecording == event.Subject {
			r.activeRecording = ""
		}
	case "progress":
		if r.completed[event.Subject] == nil {
			r.completed[event.Subject] = map[string]bool{}
		}
		r.completed[event.Subject][event.Step] = true
	case "log":
		if event.Level == "error" {
			r.failure = event.Message
		}
	}
}

// record writes what an event adds to the log. A run silences MNE's narration,
// so this trail of finished steps is all the log has to show while stages run.
func (r *liveRun) record(log *logView, event eegtable.Event) {
	switch event.Event {
	case "progress":
		log.add(r.stepLine(event))
	case "subject_start":
		if r.recording == "" {
			log.mark(event.Subject)
		}
	case "stderr", "stdout":
		log.add(event.Message)
	case "log":
		if event.Level == "error" {
			log.add(styles.Fail.Render(event.Message))
		} else {
			log.add(event.Message)
		}
	case "error":
		log.add(styles.Fail.Render(event.Message))
	}
}

// stepLine reads "[ 6/27] ✓ detect-bads", naming the recording only when the
// run covers more than one. A review step is a gate reached, not work finished.
func (r *liveRun) stepLine(event eegtable.Event) string {
	glyph := styles.Value.Render(styles.CheckMark)
	if strings.HasPrefix(event.Step, "review-") {
		glyph = styles.Strong.Render(styles.ActiveMark)
	}
	line := styles.Faint.Render(counter(event.Current, event.Total)) + " " + glyph + " " + styles.Value.Render(event.Step)
	if r.recording == "" && event.Subject != "" {
		line += styles.Dim.Render("  " + event.Subject)
	}
	return line
}

func counter(current, total int) string {
	if total <= 0 {
		return ""
	}
	return fmt.Sprintf("[%*d/%d]", len(itoa(total)), current, total)
}

func (r *liveRun) isRunning(label string) bool {
	return r != nil && r.activeRecording == label
}

func (r *liveRun) done(label, stage string) bool {
	return r != nil && r.completed[label][stage]
}

func itoa(n int) string { return strconv.Itoa(n) }

func orAll(recording string) string {
	if recording == "" {
		return "all"
	}
	return recording
}

func hint(key, label string) styles.Hint  { return styles.Hint{Key: key, Label: label, Primary: true} }
func minor(key, label string) styles.Hint { return styles.Hint{Key: key, Label: label} }

func trimPrefix(s, prefix string) string { return strings.TrimPrefix(s, prefix) }

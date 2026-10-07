//go:build windows

package eegtable

import (
	"os/exec"
	"syscall"
)

// newGroup keeps the console's Ctrl+C, which the TUI handles itself, away from
// Python, as the process group does on Unix.
func newGroup(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{CreationFlags: syscall.CREATE_NEW_PROCESS_GROUP}
}

// stopGroup ends Python alone: Windows has no signal for a whole group, so
// workers Python spawned are not reached.
func stopGroup(cmd *exec.Cmd) {
	cmd.Process.Kill()
}

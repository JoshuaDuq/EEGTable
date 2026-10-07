//go:build !windows

package eegtable

import (
	"os/exec"
	"syscall"
)

// newGroup starts the run in a process group of its own, so Stop reaches the
// workers Python may spawn.
func newGroup(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
}

func stopGroup(cmd *exec.Cmd) {
	syscall.Kill(-cmd.Process.Pid, syscall.SIGTERM)
}

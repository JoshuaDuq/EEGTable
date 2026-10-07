package styles

import (
	"os"
	"sync"

	"github.com/charmbracelet/lipgloss"
	"github.com/muesli/termenv"
)

// Theme: Monochrome Research Terminal.
// Pure grayscale palette. All emphasis is communicated through weight (bold
// vs. regular) and value (bright vs. dim), never hue. The only colored tones
// are reserved for semantically critical states that would be unsafe to
// convey through typography alone — Warning (amber) and Error (muted red).
// Everything else — Primary/Accent/Success/Info — resolves to grayscale,
// deliberately neutralizing the previous steel-blue accent.
var (
	// "Primary" is the brightest foreground; used with Bold to mark focus
	// and active selection. Selection is recognized by weight, not hue.
	// Cooled slightly (neutral off-white with a hint of blue) for a more
	// clinical, research-grade feel than the previous warm near-white.
	Primary = lipgloss.Color("#F2F3F5")
	// Used for subtle structural accents (thin bar indicators, small rules).
	Secondary = lipgloss.Color("#34363A")
	// A dimmer off-white used for secondary emphasis. Sits between Text
	// and Muted so accented values feel deliberate without shouting.
	Accent = lipgloss.Color("#BFC3C7")
	// Success resolves to the same bright off-white as Primary: meaning is
	// carried by the ✓ mark, not by color.
	Success = lipgloss.Color("#F2F3F5")
	// Warning and Error are desaturated so they remain distinguishable in
	// monochrome contexts and for users with custom palettes — tuned to
	// feel considered rather than alarming.
	Warning = lipgloss.Color("#C49A5B")
	Error   = lipgloss.Color("#C07878")
	// Info is grayscale; informational highlights rely on Bold instead of
	// a colored accent.
	Info         = lipgloss.Color("#BFC3C7")
	Muted        = lipgloss.Color("#6A6D72") // De-emphasized text.
	Text         = lipgloss.Color("#E6E8EB") // Primary text — cool near-white.
	TextDim      = lipgloss.Color("#A4A8AD") // Secondary text.
	BgDark       = lipgloss.Color("#0A0B0D") // App background — inky black.
	Surface      = lipgloss.Color("#131518") // Card/panel background.
	SurfaceAlt   = lipgloss.Color("#1A1C20") // Slightly-raised panel surface.
	Border       = lipgloss.Color("#3A3D43") // Subtle panel borders.
	BorderBright = lipgloss.Color("#585B61") // Focus-adjacent borders.
)

var noColorOnce sync.Once

// ApplyNoColorProfile respects the NO_COLOR environment variable (https://no-color.org).
// Safe to call repeatedly; only applies the first time.
func ApplyNoColorProfile() {
	noColorOnce.Do(func() {
		if v := os.Getenv("NO_COLOR"); v != "" {
			// termenv numbers its profiles from TrueColor, so a literal 0 forced full colour.
			lipgloss.SetColorProfile(termenv.Ascii)
		}
	})
}

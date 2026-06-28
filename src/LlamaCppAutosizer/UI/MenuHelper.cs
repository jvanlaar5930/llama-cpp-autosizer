using Spectre.Console;

namespace LlamaCppAutosizer.UI;

/// <summary>
/// Lightweight arrow-key navigation menu that supports Escape to go back.
/// Replaces Spectre's SelectionPrompt so Escape and Ctrl+C work everywhere
/// without racing against Spectre's internal Console.ReadKey loop.
/// </summary>
public static class MenuHelper
{
    // Returns the chosen item, or null when Escape is pressed or ct is cancelled.
    public static string? Select(
        string title,
        IEnumerable<string> choices,
        int pageSize = 15,
        CancellationToken ct = default)
    {
        var items = choices.ToList();
        if (items.Count == 0) return null;

        int selected = 0;
        int scrollOffset = 0;
        int menuStartRow = 0;
        int menuHeight = 0;

        Console.CursorVisible = false;
        try
        {
            // ── Pre-scroll so the menu fits below the current cursor ────────────
            // title + separator + items + scroll-info + footer = visible + 4 lines
            int visible = Math.Min(pageSize, items.Count);
            int needed = visible + 4;

            // WindowHeight can be 0 in non-interactive contexts; guard against it.
            int windowH = Math.Max(1, Console.WindowHeight);
            needed = Math.Min(needed, windowH - 1);

            int gap = windowH - Console.CursorTop;
            if (gap < needed)
            {
                // Print blank lines to push the menu into view, then move back up.
                int pushDown = needed - gap;
                for (int i = 0; i < pushDown; i++) Console.WriteLine();
                Console.SetCursorPosition(0, Math.Max(0, Console.CursorTop - pushDown));
            }
            menuStartRow = Console.CursorTop;

            // ── Render + input loop ─────────────────────────────────────────────
            bool firstRender = true;
            while (!ct.IsCancellationRequested)
            {
                // Clamp scroll window
                visible = Math.Min(pageSize, items.Count);
                if (selected < scrollOffset) scrollOffset = selected;
                if (selected >= scrollOffset + visible) scrollOffset = selected - visible + 1;

                if (!firstRender)
                    SafeSetCursor(0, menuStartRow);
                firstRender = false;

                RenderMenu(title, items, selected, scrollOffset, visible, out menuHeight);

                // Poll for key in short bursts so Ctrl+C can cancel us without
                // being stuck behind a blocking Console.ReadKey call.
                while (!Console.KeyAvailable)
                {
                    if (ct.IsCancellationRequested)
                    {
                        SafeSetCursor(0, menuStartRow + menuHeight);
                        return null;
                    }
                    Thread.Sleep(20);
                }

                var key = Console.ReadKey(intercept: true);
                switch (key.Key)
                {
                    case ConsoleKey.UpArrow:
                        selected = selected > 0 ? selected - 1 : items.Count - 1;
                        break;

                    case ConsoleKey.DownArrow:
                        selected = selected < items.Count - 1 ? selected + 1 : 0;
                        break;

                    case ConsoleKey.Home:
                        selected = 0;
                        break;

                    case ConsoleKey.End:
                        selected = items.Count - 1;
                        break;

                    case ConsoleKey.PageUp:
                        selected = Math.Max(0, selected - pageSize);
                        break;

                    case ConsoleKey.PageDown:
                        selected = Math.Min(items.Count - 1, selected + pageSize);
                        break;

                    case ConsoleKey.Enter:
                        SafeSetCursor(0, menuStartRow + menuHeight);
                        Console.WriteLine();
                        return items[selected];

                    case ConsoleKey.Escape:
                        SafeSetCursor(0, menuStartRow + menuHeight);
                        Console.WriteLine();
                        return null;
                }
            }

            // Loop exited because ct was cancelled
            SafeSetCursor(0, menuStartRow + menuHeight);
            return null;
        }
        finally
        {
            Console.CursorVisible = true;
        }
    }

    // Overload for (label, value) pairs — returns the value or null on Escape.
    public static T? Select<T>(
        string title,
        IEnumerable<(string Label, T Value)> choices,
        int pageSize = 15) where T : class
    {
        var list = choices.ToList();
        var chosen = Select(title, list.Select(c => c.Label), pageSize);
        if (chosen is null) return null;
        int idx = list.FindIndex(c => c.Label == chosen);
        return idx >= 0 ? list[idx].Value : null;
    }

    // -------------------------------------------------------------------------
    // Rendering
    // -------------------------------------------------------------------------

    private static void RenderMenu(
        string title, List<string> items,
        int selected, int scrollOffset, int visible,
        out int linesWritten)
    {
        int w = Math.Max(20, Console.WindowWidth);
        int lines = 0;

        void WriteLine(string markup)
        {
            AnsiConsole.MarkupLine(ClearToEol(markup, w));
            lines++;
        }

        WriteLine(title);
        WriteLine("[grey]──────────────────────────────────────────────────────[/]");

        for (int i = scrollOffset; i < scrollOffset + visible; i++)
        {
            bool isSelected = i == selected;
            string raw = items[i];
            string content = isSelected
                ? $"[bold cyan] > {raw}[/]"
                : $"   {raw}";
            WriteLine(ClearToEol(content, w));
        }

        // Scroll info or blank spacer — keeps the footer row stable
        if (items.Count > visible)
            WriteLine($"[grey] ({scrollOffset + 1}–{scrollOffset + visible} of {items.Count})[/]");
        else
            WriteLine(ClearToEol("", w));

        WriteLine("[grey] ↑ ↓ Navigate   Enter Select   Esc Back[/]");

        linesWritten = lines;
    }

    // Pads to terminal width so re-renders don't leave ghost characters from longer previous lines.
    private static string ClearToEol(string markup, int width)
    {
        string plain = System.Text.RegularExpressions.Regex.Replace(markup, @"\[.*?\]", "");
        int pad = Math.Max(0, width - plain.Length - 1);
        return markup + new string(' ', pad);
    }

    // SetCursorPosition throws if the row is >= BufferHeight; clamp defensively.
    private static void SafeSetCursor(int left, int top)
    {
        try
        {
            int maxRow = Math.Max(0, Console.BufferHeight - 1);
            Console.SetCursorPosition(left, Math.Min(top, maxRow));
        }
        catch (ArgumentOutOfRangeException) { /* ignore in edge cases */ }
    }
}

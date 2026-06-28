using Spectre.Console;

namespace LlamaCppAutosizer.UI;

/// <summary>
/// Lightweight arrow-key navigation menu with Escape-to-back and Ctrl+C support.
/// Uses relative ANSI cursor-up movement for re-renders so the menu stays in place
/// even when the terminal scrolls — absolute row tracking breaks after a scroll.
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
        int lastMenuHeight = 0;

        Console.CursorVisible = false;
        try
        {
            // Pre-scroll: if there isn't enough room below the cursor for the menu,
            // print blank lines to push the content down, then move the cursor back up.
            int visible = Math.Min(pageSize, items.Count);
            int needed = visible + 4; // title + separator + items + scroll-info + footer
            int windowH = Math.Max(8, Console.WindowHeight);
            needed = Math.Min(needed, windowH - 1);
            int gap = windowH - Console.CursorTop;
            if (gap < needed)
            {
                int pushDown = needed - gap;
                for (int i = 0; i < pushDown; i++) Console.WriteLine();
                // Move cursor back up using relative ANSI (not SetCursorPosition)
                Console.Write($"\x1b[{pushDown}A\r");
            }

            bool firstRender = true;

            while (!ct.IsCancellationRequested)
            {
                // Clamp scroll window
                visible = Math.Min(pageSize, items.Count);
                if (selected < scrollOffset) scrollOffset = selected;
                if (selected >= scrollOffset + visible) scrollOffset = selected - visible + 1;

                // On re-renders, move cursor UP by the number of lines we wrote last time.
                // This is relative and survives terminal scrolling unlike SetCursorPosition.
                if (!firstRender && lastMenuHeight > 0)
                    Console.Write($"\x1b[{lastMenuHeight}A\r");
                firstRender = false;

                RenderMenu(title, items, selected, scrollOffset, visible, out lastMenuHeight);

                // Poll for a key in short intervals so Ctrl+C can cancel without blocking.
                while (!Console.KeyAvailable)
                {
                    if (ct.IsCancellationRequested)
                        return null;
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
                        Console.WriteLine();
                        return items[selected];
                    case ConsoleKey.Escape:
                        Console.WriteLine();
                        return null;
                }
            }

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
            string raw = items[i];
            string content = i == selected
                ? $"[bold cyan] > {raw}[/]"
                : $"   {raw}";
            WriteLine(ClearToEol(content, w));
        }

        // Scroll indicator or blank spacer — keeps footer on a stable row
        if (items.Count > visible)
            WriteLine($"[grey] ({scrollOffset + 1}–{scrollOffset + visible} of {items.Count})[/]");
        else
            WriteLine(ClearToEol("", w));

        WriteLine("[grey] ↑ ↓ Navigate   Enter Select   Esc Back[/]");

        linesWritten = lines;
    }

    // Pads to terminal width so re-renders erase any leftover characters from longer lines.
    private static string ClearToEol(string markup, int width)
    {
        string plain = System.Text.RegularExpressions.Regex.Replace(markup, @"\[.*?\]", "");
        int pad = Math.Max(0, width - plain.Length - 1);
        return markup + new string(' ', pad);
    }
}

using System;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Markup.Xaml;
using Avalonia.VisualTree;

namespace Vernacula.Tts.App.Views;

public partial class MainWindow : Window
{
    public MainWindow() => InitializeComponent();

    private void InitializeComponent() => AvaloniaXamlLoader.Load(this);

    // The type-ahead pickers: focusing one opens its full list and selects its text, so the next
    // keystroke starts a fresh search instead of appending to the current choice's name.
    private void OnPickerGotFocus(object? sender, GotFocusEventArgs e)
    {
        if (sender is not AutoCompleteBox box) return;
        box.FindDescendantOfType<TextBox>()?.SelectAll();
        box.IsDropDownOpen = true;
    }

    protected override void OnClosed(EventArgs e)
    {
        // The VM owns the synthesis service (ORT sessions, several GB of
        // model weights) and the playback service (potentially a live
        // ffplay subprocess). Process exit reclaims memory, but a forced
        // ffplay child outliving the GUI is a real failure mode — dispose
        // explicitly on window close.
        (DataContext as IDisposable)?.Dispose();
        base.OnClosed(e);
    }
}

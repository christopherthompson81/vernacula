using System;
using Avalonia.Controls;
using Avalonia.Markup.Xaml;

namespace Chatterbox.App.Views;

public partial class MainWindow : Window
{
    public MainWindow() => InitializeComponent();

    private void InitializeComponent() => AvaloniaXamlLoader.Load(this);

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

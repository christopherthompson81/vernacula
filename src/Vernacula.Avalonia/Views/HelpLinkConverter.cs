using System.Windows.Input;
using Avalonia.Controls;
using Vernacula.App.Services;

namespace Vernacula.App.Views;

public class HelpLinkConverter : ICommand
{
    public static HelpLinkConverter Instance { get; } = new();
    public static HelpWindow? CurrentWindow { get; set; }

    public event EventHandler? CanExecuteChanged
    {
        add { }
        remove { }
    }

    public bool CanExecute(object? parameter) => true;

    public void Execute(object? parameter)
    {
        if (parameter is not string url)
            return;

        if (url.StartsWith("http", StringComparison.OrdinalIgnoreCase))
        {
            System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo { FileName = url, UseShellExecute = true });
            return;
        }

        var helpWindow = CurrentWindow;
        if (helpWindow == null)
            return;

        var current = helpWindow.CurrentTopicId ?? "index";
        var currentTopic = HelpService.FindById(current) ?? HelpService.IndexTopic;
        var target = HelpService.ResolveRelativeLink(currentTopic, url);
        if (target != null)
            helpWindow.DisplayTopic(target.TopicId);
    }
}
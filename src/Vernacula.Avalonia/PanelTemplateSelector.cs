using Vernacula.App.ViewModels;

namespace Vernacula.App;

// NOTE: Avalonia handles template selection differently than WPF.
// This class is kept for compatibility but template selection is now handled
// through ContentControl with direct bindings in the XAML.
public class PanelTemplateSelector
{
    public object? HomeTemplate     { get; set; }
    public object? ProgressTemplate { get; set; }
    public object? ResultsTemplate  { get; set; }

    // This method is not used - template selection is done in XAML
    internal object? SelectTemplate(AppPanel panel) => null;
}

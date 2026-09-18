using System.Collections.ObjectModel;
using Avalonia.Media;
using CommunityToolkit.Mvvm.ComponentModel;

namespace Vernacula.App.ViewModels;

/// <summary>
/// One row of a table card. The cells are dense — a grid position the markdown left empty still
/// gets a cell, holding no words — because the row is drawn by a panel that places children in
/// order, and a sparse row would slide its later cells under the wrong headings.
/// </summary>
public sealed class TableRowViewModel
{
    public TableRowViewModel(int index, bool isHeader, int columnCount)
    {
        Index = index;
        IsHeader = isHeader;
        ColumnCount = columnCount;
    }

    public int Index { get; }
    public bool IsHeader { get; }

    /// <summary>Columns in the whole table, not in this row — what makes the columns line up.</summary>
    public int ColumnCount { get; }

    public ObservableCollection<TableCellViewModel> Cells { get; } = new();
}

/// <summary>
/// One cell of a table card: the words the extractor put in this grid position, in spoken order,
/// plus the display order the panel needs where the two differ (see
/// <see cref="BlockItemViewModel.UpdateFlowDirection"/>).
///
/// <para>The words are the SAME objects the block holds in <see cref="BlockItemViewModel.Words"/>
/// — the cell is another view of them, not a copy — so highlighting a word as the audio reaches it
/// lights it up wherever it is drawn.</para>
/// </summary>
public sealed partial class TableCellViewModel : ObservableObject
{
    public TableCellViewModel(int row, int column, bool isHeader)
    {
        Row = row;
        Column = column;
        IsHeader = isHeader;
    }

    public int Row { get; }
    public int Column { get; }
    public bool IsHeader { get; }

    public ObservableCollection<WordItemViewModel> Words { get; } = new();
    public ObservableCollection<WordItemViewModel> Display { get; } = new();

    [ObservableProperty] private FlowDirection _flowDirection = FlowDirection.LeftToRight;
}

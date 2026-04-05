using Avalonia;

namespace Vernacula.Avalonia.Extensions;

/// <summary>
/// Extension methods for visual tree traversal.
/// TODO: Port to Avalonia - Parent property not available
/// </summary>
// public static class VisualExtensions
// {
//     /// <summary>
//     /// Gets the first visual parent of the specified type.
//     /// </summary>
//     public static T? GetVisualParent<T>(this AvaloniaObject element) where T : class
//     {
//         var parent = element.Parent;
//         while (parent != null)
//         {
//             if (parent is T t)
//                 return t;
//             parent = parent.Parent;
//         }
//         return null;
//     }
// }

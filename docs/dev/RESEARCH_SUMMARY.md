# Research Summary: Avalonia 11.x Trigger Mechanism

## Executive Summary

I've completed comprehensive research on Avalonia 11.x's alternative trigger mechanism for implementing hover/pressed states. The key finding is that **Avalonia 11.x uses CSS-like pseudo-class selectors** instead of WPF-style triggers, representing a fundamental architectural change.

## Key Discovery

### WPF/Old Avalonia (NOT Supported in 11.x)
```xml
<ControlTemplate.Triggers>
    <Trigger Property="IsMouseOver" Value="True">
        <Setter Property="Background" Value="Blue" />
    </Trigger>
</ControlTemplate.Triggers>
```

### Avalonia 11.x (NEW Approach)
```xml
<Style Selector="Button:pointerover">
    <Setter Property="Background" Value="Blue" />
</Style>
<Style Selector="Button:pressed">
    <Setter Property="Background" Value="DarkBlue" />
</Style>
```

## Available Pseudo-Classes

| Selector | Description | Auto-Applied |
|----------|-------------|--------------|
| `:pointerover` | Mouse hover | ✅ Yes |
| `:pressed` | Mouse button held | ✅ Yes |
| `:focus` | Keyboard focus | ✅ Yes |
| `:disabled` | Control disabled | ✅ Yes |
| `:checked` | Toggle checked | ✅ Yes |
| `:indeterminate` | Toggle indeterminate | ✅ Yes |
| `:valid` | Validation valid | ✅ Yes |
| `:invalid` | Validation invalid | ✅ Yes |

## WPF Properties That Don't Exist in Avalonia 11.x

| WPF Property | Avalonia Equivalent | Status |
|--------------|-------------------|--------|
| `StrokeLineJoin` | `StrokeJoin` | ❌ Must Replace |
| `ContentSource="X"` | `Content="{Binding X, RelativeSource={RelativeSource TemplatedParent}}"` | ❌ Must Replace |
| `ShadowDepth` + `Direction` | Remove (not supported in same way) | ❌ Must Remove |
| `AllowsTransparency` | (remove - not needed) | ❌ Must Remove |
| `PopupAnimation` | (remove - not supported) | ❌ Must Remove |
| `Visibility="Collapsed"` on shapes | `IsVisible="False"` | ⚠️ Should Replace |
| `SnapsToDevicePixels` on ItemsPresenter | (remove - not supported) | ⚠️ Should Remove |
| `KeyboardNavigation.DirectionalNavigation` on ItemsPresenter | (remove - not supported) | ⚠️ Should Remove |

## Current Project Status

**Total Build Errors:** 277
- CommonStyles.axaml: ~85 errors (WPF properties + triggers)
- Other XAML files: ~192 errors (various issues)

**Primary Issues in CommonStyles.axaml:**
1. `ControlTemplate.Triggers` blocks (removed but no replacement)
2. `Style.Triggers` blocks (removed but no replacement)
3. WPF-specific properties still present
4. No pseudo-class styles implemented

## Documentation Created

I've created 4 comprehensive documentation files:

### 1. `docs/AVALONIA_PSEUDO_CLASSES_GUIDE.md`
**Purpose:** Complete reference for Avalonia 11.x pseudo-class system

**Contents:**
- Overview of pseudo-class mechanism
- Complete list of built-in pseudo-classes
- Custom pseudo-class implementation (C#)
- Detailed examples for all control types:
  - Button hover/pressed states
  - ComboBox states
  - MenuItem states
  - DataGrid row/cell states
  - TextBox focus states
  - CheckBox/RadioButton states
- WPF property replacement guide
- Selector syntax reference
- Migration checklist

### 2. `docs/COMMONSTYLES_MIGRATION_EXAMPLES.md`
**Purpose:** Before/after examples specific to your CommonStyles.axaml

**Contents:**
- Exact code changes for each control type
- 6 detailed migration examples:
  1. Base Button Style
  2. Sidebar Topic Button
  3. MenuItem (with all WPF property fixes)
  4. ComboBox (with all WPF property fixes)
  5. DataGrid Row/Cell
  6. ScrollBar Thumb
- Summary table of property replacements
- Testing checklist

### 3. `docs/AVALONIA_TRIGGER_MIGRATION_SUMMARY.md`
**Purpose:** Executive summary and implementation strategy

**Contents:**
- Key findings summary
- Comparison table (WPF vs Avalonia)
- Current build error analysis
- 3-phase implementation strategy:
  - Phase 1: Fix WPF properties (critical)
  - Phase 2: Add pseudo-class styles
  - Phase 3: Add new brushes
- Technical notes on selector syntax
- Resources and references

### 4. `docs/AVALONIA_QUICK_REFERENCE.md`
**Purpose:** Quick lookup card for common replacements

**Contents:**
- Common state replacement table
- Common property replacement table
- Shadow direction conversion guide
- Quick style patterns (copy/paste ready)
- Selector combinator reference
- Common issues (what doesn't work vs what does)
- Testing checklist

## Implementation Recommendations

### Recommended Approach: Incremental Migration

**Phase 1: Fix WPF-Specific Properties (Critical)**
These changes are required just to get the project building:

1. Replace `StrokeLineJoin` → `StrokeJoin`
2. Replace `ContentSource="X"` → explicit binding
3. Replace `ShadowDepth`/`Direction` → remove DropShadowEffect
4. Remove `AllowsTransparency` from Popup
5. Remove `PopupAnimation` from Popup
6. Replace `Visibility="Collapsed"` → `IsVisible="False"` on shapes
7. Remove `SnapsToDevicePixels` from ItemsPresenter
8. Remove `KeyboardNavigation.DirectionalNavigation` from ItemsPresenter

**Phase 2: Add Pseudo-Class Styles (Visual Feedback)**
After the project builds, add hover/pressed states:

```xml
<!-- Button states -->
<Style Selector="Button:pointerover">
    <Setter Property="Background" Value="{DynamicResource OverlayBrush}" />
</Style>
<Style Selector="Button:pressed">
    <Setter Property="Background" Value="{DynamicResource AccentBrush}" />
</Style>

<!-- ComboBox states -->
<Style Selector="ComboBox:pointerover">
    <Setter Property="BorderBrush" Value="{DynamicResource AccentBrush}" />
</Style>
<Style Selector="ComboBoxItem:pointerover">
    <Setter Property="Background" Value="{DynamicResource OverlayBrush}" />
</Style>

<!-- DataGrid states -->
<Style Selector="DataGridRow:pointerover">
    <Setter Property="Background" Value="{DynamicResource OverlayBrush}" />
</Style>
<Style Selector="DataGridRow:selected">
    <Setter Property="Background" Value="{DynamicResource AccentBrush}" />
    <Setter Property="Foreground" Value="White" />
</Style>
```

**Phase 3: Enhance with Additional Brushes**
Add hover/pressed color variations to theme files for refined visuals.

### Alternative Approach: Simplify First

If you need the app running immediately:
1. Comment out all complex ControlTemplates
2. Let Avalonia use default control templates
3. Add simple pseudo-class styles for basic feedback
4. Gradually restore custom templates

## Technical Details

### Pseudo-Class Selector Syntax

```xml
<!-- Basic -->
<Style Selector="ControlType:pseudo-class">
    <Setter Property="Property" Value="Value" />
</Style>

<!-- With classes -->
<Style Selector="ControlType.my-class:pointerover">
    <Setter Property="Background" Value="Blue" />
</Style>

<!-- Targeting template parts -->
<Style Selector="ControlType:pointerover > Border#PartName">
    <Setter Property="Background" Value="Blue" />
</Style>

<!-- Combined states -->
<Style Selector="DataGridRow:selected:pointerover">
    <Setter Property="Background" Value="DarkBlue" />
</Style>
```

### Custom Pseudo-Classes (C#)

```csharp
// Add custom state
myControl.PseudoClasses.Add(":custom");

// Remove custom state
myControl.PseudoClasses.Remove(":custom");
```

Then in XAML:
```xml
<Style Selector="ControlType:custom">
    <Setter Property="Background" Value="Red" />
</Style>
```

## Shadow Effect Note

**Important:** Avalonia's `DropShadowEffect` doesn't have the same API as WPF:
- No `Offset` property (uses different mechanism)
- No `ShadowDepth` or `Direction` properties

**Recommendation:** Remove DropShadowEffect entirely for now. Avalonia's default rendering provides adequate visual depth, or consider using `BoxShadow` from a third-party package if shadows are critical.

## Next Steps

Based on this research, you have several options:

### Option 1: Systematic Migration (Recommended)
Use the documentation to methodically fix properties and add pseudo-class styles. This provides:
- ✅ Full control over styling
- ✅ Best visual results
- ✅ Future-proof code
- ⏱️ Requires time investment

### Option 2: Quick Fix
Comment out problematic templates and use Avalonia defaults:
- ✅ Fast implementation
- ✅ App builds and runs
- ❌ Less visual polish
- ❌ Missing hover/pressed feedback initially

### Option 3: Hybrid Approach
Fix critical WPF properties first to get the build working, then add pseudo-class styles incrementally:
- ✅ Balanced approach
- ✅ Immediate progress
- ✅ Can test as you go

## Resources

### Documentation Files Created
- `docs/AVALONIA_PSEUDO_CLASSES_GUIDE.md` - Complete reference
- `docs/COMMONSTYLES_MIGRATION_EXAMPLES.md` - Project-specific examples
- `docs/AVALONIA_TRIGGER_MIGRATION_SUMMARY.md` - Strategy and summary
- `docs/AVALONIA_QUICK_REFERENCE.md` - Quick lookup

### External Resources
- [Avalonia Styling Documentation](https://docs.avaloniaui.net/docs/controls/styling)
- [Avalonia Pseudo-Classes](https://docs.avaloniaui.net/docs/controls/styling/pseudo-classes)
- [Avalonia Selectors](https://docs.avaloniaui.net/docs/controls/styling/selectors)

## Conclusion

The migration from WPF-style triggers to Avalonia's pseudo-class system is **straightforward but requires systematic changes**. The new system is actually **more intuitive** (CSS-like) and **more powerful** (combinators, custom pseudo-classes).

**Key Takeaway:** Avalonia 11.x doesn't support triggers in the same way as WPF. Instead, it uses a modern CSS-like pseudo-class selector system that provides the same functionality with a cleaner syntax.

Would you like me to proceed with implementing the fixes based on this research?

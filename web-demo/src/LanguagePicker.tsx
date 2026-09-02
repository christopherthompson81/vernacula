import { useEffect, useMemo, useRef, useState } from "react";
import { LANGUAGES, type LanguageOption } from "./inference/languages.ts";
import { search } from "./langSearch.ts";

/**
 * Type-ahead language picker.
 *
 * ⚠ A `<select>` does not scale to 193 rows. Browsers give a native select only first-letter
 * matching, so finding "Xhosa" means scrolling past 180 entries, and the codes — which are how
 * anyone who knows what they want will search — are not matched at all. The ranking lives in
 * langSearch.ts so tools/check-language-search.mjs can verify it over all 193.
 */

export function LanguagePicker({ value, disabled, onChange }: {
  value: string; disabled?: boolean; onChange: (code: string) => void;
}) {
  const current = LANGUAGES.find((l) => l.code === value);
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const box = useRef<HTMLDivElement | null>(null);
  const listRef = useRef<HTMLUListElement | null>(null);

  const matches = useMemo(() => search(LANGUAGES, query), [query]);

  useEffect(() => { setActive(0); }, [query]);
  useEffect(() => {
    if (!open) return;
    const away = (e: MouseEvent) => { if (!box.current?.contains(e.target as Node)) setOpen(false); };
    document.addEventListener("mousedown", away);
    return () => document.removeEventListener("mousedown", away);
  }, [open]);
  // Keep the keyboard cursor visible without scrolling the page itself.
  useEffect(() => {
    listRef.current?.children[active]?.scrollIntoView({ block: "nearest" });
  }, [active, open]);

  const choose = (l: LanguageOption) => { onChange(l.code); setQuery(""); setOpen(false); };

  return (
    <div className="picker" ref={box}>
      <input
        type="text" role="combobox" aria-expanded={open} aria-controls="lang-list"
        aria-autocomplete="list"
        aria-activedescendant={open && matches[active] ? `lang-opt-${matches[active].code}` : undefined}
        disabled={disabled} value={open ? query : (current?.name ?? value)}
        placeholder="Search 193 languages…"
        onFocus={(e) => { setOpen(true); setQuery(""); e.currentTarget.select(); }}
        onChange={(e) => { setQuery(e.target.value); setOpen(true); }}
        onKeyDown={(e) => {
          if (e.key === "ArrowDown") { setOpen(true); setActive((i) => Math.min(i + 1, matches.length - 1)); e.preventDefault(); }
          else if (e.key === "ArrowUp") { setActive((i) => Math.max(i - 1, 0)); e.preventDefault(); }
          else if (e.key === "Enter" && open && matches[active]) { choose(matches[active]); e.preventDefault(); }
          else if (e.key === "Escape") { setOpen(false); (e.target as HTMLInputElement).blur(); }
        }} />
      {open && (
        <ul className="options" id="lang-list" role="listbox" ref={listRef}>
          {matches.length === 0 && <li className="empty">no language matches “{query}”</li>}
          {matches.map((l, i) => (
            <li key={l.code} id={`lang-opt-${l.code}`} role="option" aria-selected={l.code === value}
                className={i === active ? "active" : undefined}
                onMouseEnter={() => setActive(i)}
                onMouseDown={(e) => { e.preventDefault(); choose(l); }}>
              <span className="nm">{l.name}</span>
              {/* ⚠ dir="auto" — an Arabic or Hebrew endonym in an otherwise LTR row renders its
                  punctuation and any adjacent digits on the wrong side without it. */}
              {l.native && <span className="nat" dir="auto" lang={l.code}>{l.native}</span>}
              <span className="cd">{l.code}</span>
              {l.trained && <span className="badge trained" title="in the fine-tune corpus">trained</span>}
              {l.voice && <span className="badge" title={`no native reference voice — read by the ${l.voice} voice`}>voice {l.voice}</span>}
              <span className="sz">{l.mb >= 0.1 ? `${l.mb.toFixed(l.mb >= 10 ? 0 : 1)} MB` : ""}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

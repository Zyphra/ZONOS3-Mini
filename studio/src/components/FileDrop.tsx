import { useEffect, useRef, useState } from 'preact/hooks';
import { Icon } from './Icon';

interface Props {
  /** What a drop does, e.g. “Drop to clone this voice”. */
  label: string;
  hint?: string;
  icon?: string;
  /** Shown instead of the label while a drop can't be taken (the overlay still stays, so the drop isn't lost to the browser). */
  busy?: string | null;
  onFile: (f: File) => void;
}

const hasFiles = (e: DragEvent) => !!e.dataTransfer?.types.includes('Files');

/**
 * The whole window takes a dropped file: from the moment files are dragged over the page until they are dropped or
 * dragged away, a full-window overlay says what the drop will do. Mount it on the screen that owns the drop.
 */
export function FileDrop({ label, hint, icon = 'upload', busy, onFile }: Props) {
  const [over, setOver] = useState(false);
  // Every dragenter (on the window or any element inside) pairs with a dragleave; the page is left at zero.
  const depth = useRef(0);
  const take = useRef({ busy, onFile });
  take.current = { busy, onFile };
  useEffect(() => {
    const enter = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      depth.current++;
      setOver(true);
    };
    const leave = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      depth.current = Math.max(0, depth.current - 1);
      if (!depth.current) setOver(false);
    };
    const overPage = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      e.dataTransfer!.dropEffect = take.current.busy ? 'none' : 'copy';
    };
    const drop = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      depth.current = 0;
      setOver(false);
      const f = e.dataTransfer!.files[0];
      if (f && !take.current.busy) take.current.onFile(f);
    };
    window.addEventListener('dragenter', enter);
    window.addEventListener('dragleave', leave);
    window.addEventListener('dragover', overPage);
    window.addEventListener('drop', drop);
    return () => {
      window.removeEventListener('dragenter', enter);
      window.removeEventListener('dragleave', leave);
      window.removeEventListener('dragover', overPage);
      window.removeEventListener('drop', drop);
    };
  }, []);
  if (!over) return null;
  return (
    <div class={`file-drop${busy ? ' is-busy' : ''}`} aria-hidden="true">
      <div class="file-drop-card">
        <Icon name={icon} size={30} />
        <b>{busy ?? label}</b>
        {hint && !busy && <span class="muted small">{hint}</span>}
      </div>
    </div>
  );
}

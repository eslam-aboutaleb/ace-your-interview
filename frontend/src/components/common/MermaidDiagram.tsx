import { useEffect, useId, useState } from "react";
import mermaid from "mermaid";
import CodePanel from "@/components/common/CodePanel";

interface MermaidDiagramProps {
  code: string;
  compact?: boolean;
}

let mermaidConfigured = false;

function ensureMermaidConfigured() {
  if (mermaidConfigured) return;
  mermaid.initialize({
    startOnLoad: false,
    securityLevel: "strict",
    theme: "default",
    suppressErrorRendering: true,
  });
  mermaidConfigured = true;
}

export default function MermaidDiagram({ code, compact = false }: MermaidDiagramProps) {
  const [svg, setSvg] = useState("");
  const [renderError, setRenderError] = useState(false);
  const rawId = useId();
  const id = `mermaid-${rawId.replace(/[^a-zA-Z0-9_-]/g, "")}`;

  useEffect(() => {
    let active = true;
    const renderDiagram = async () => {
      ensureMermaidConfigured();
      try {
        const rendered = await mermaid.render(id, code);
        if (!active) return;
        setSvg(rendered.svg);
        setRenderError(false);
      } catch {
        if (!active) return;
        setSvg("");
        setRenderError(true);
      }
    };
    renderDiagram();
    return () => {
      active = false;
    };
  }, [code, id]);

  if (renderError) {
    return (
      <div className="diagram-fallback">
        <p className="diagram-fallback-title">Diagram rendering failed. Showing source:</p>
        <CodePanel code={code} language="mermaid" compact={compact} />
      </div>
    );
  }

  if (!svg) {
    return (
      <div className="diagram-panel">
        <p className="diagram-loading">Rendering diagram...</p>
      </div>
    );
  }

  return (
    <div className="diagram-panel">
      <div className="diagram-scroll" dangerouslySetInnerHTML={{ __html: svg }} />
    </div>
  );
}

import { useEffect, useState } from "react";
import "bootstrap/dist/css/bootstrap.min.css";

const API_BASE = "";

interface AssignmentResponse {
  assignment_id: number;
  comment_id: number;
  video_title: string;
  comment: string;
  done?: boolean;
}

interface ProgressResponse {
  done: number;
  total: number;
}

const ASPECT_LABELS: Record<string, string> = {
  RELIGION: "Religião",
  POLITICS: "Política",
  GENDER_DENIAL: "Invalidação de gênero",
  HARRASSMENT: "Assédio",
  THREATENING: "Ameaça",
  PSEUDOSCIENCE: "Pseudociência",
};

const TOKEN_TTL_MINUTES = 30;
const TOKEN_CACHE_KEY = "annotation_token";
const TOKEN_EXP_KEY = "annotation_token_exp";

function GuideContent() {
  return (
    <div style={{ fontSize: "0.95rem" }}>
      <p className="mb-2">
        Durante as anotações, você irá ler comentários e classificá-los de acordo com:
      </p>

      <strong>1. Atitude em relação a pessoas trans</strong>
      <ul className="mb-3">
        <li>
          <strong>Apoio:</strong> demonstra apoio ou empatia.
          <br />
          <em>Ex.: “todo mundo merece respeito”, “o que custa demonstrar empatia?”</em>
        </li>
        <li>
          <strong>Neutro:</strong> não expressa julgamento claro ou não está relacionado ao tema.
          <br />
          <em>Ex.: “sei nem o que dizer”, “que vídeo interessante”</em>
        </li>
        <li>
          <strong>Ódio / Intolerância:</strong> expressa rejeição, invalidação ou ataque.
          <br />
          <em>Ex.: “isso é absurdo, essas pessoas não deveriam existir”, “que nojo”</em>
        </li>
      </ul>

      <strong>2. Linguagem ofensiva</strong>
      <ul className="mb-3">
        <li>
          <strong>Ofensivo:</strong> contém insultos, xingamentos ou humilhações.
          <br />
          <em>Ex.: “que gente nojenta”</em>
        </li>
        <li>
          <strong>Não ofensivo:</strong> não contém esse tipo de linguagem.
          <br />
          <em>Ex.: “não concordo com isso”</em>
        </li>
      </ul>

      <strong>3. Vertentes</strong>
      <p className="mb-1">
        Marque uma ou mais opções que melhor representem o foco do comentário:
      </p>
      <ul className="mb-3">
        <li>
          Religião — <em>“isso vai contra a vontade de deus”</em>
        </li>
        <li>
          Política — <em>“isso é culpa da esquerda”</em>
        </li>
        <li>
          Negação de gênero — <em>“mas só existe homem e mulher”</em>
        </li>
        <li>
          Assédio — <em>“essas pessoas são doentes”</em>
        </li>
        <li>
          Ameaça — <em>“se eu te encontro na rua, eu te machucaria”</em>
        </li>
        <li>
          Pseudociência — <em>“mas só existe cromossomo XX e XY”</em>
        </li>
      </ul>

      <p className="mb-0">
        Em caso de dúvida, escolha a opção que melhor represente o sentido predominante do
        comentário. Caso não encontre, não é obrigatório marcar as vertentes.
      </p>
    </div>
  );
}

interface GuideButtonProps {
  onClick: () => void;
}

function GuideButton({ onClick }: GuideButtonProps) {
  return (
    <button
      type="button"
      className="btn btn-outline-secondary btn-sm"
      onClick={onClick}
      title="Abrir guia de anotação"
    >
      Guia de anotação
    </button>
  );
}

interface GuideModalProps {
  open: boolean;
  onClose: () => void;
}

function GuideModal({ open, onClose }: GuideModalProps) {
  if (!open) return null;

  return (
    <>
      <div
        onClick={onClose}
        style={{
          position: "fixed",
          inset: 0,
          backgroundColor: "rgba(0,0,0,0.35)",
          zIndex: 1040,
        }}
      />
      <div
        className="card shadow"
        style={{
          position: "fixed",
          top: "5%",
          left: "50%",
          transform: "translateX(-50%)",
          width: "min(760px, 92vw)",
          maxHeight: "85vh",
          overflowY: "auto",
          zIndex: 1050,
        }}
      >
        <div className="card-header d-flex justify-content-between align-items-center">
          <strong>Guia de anotação</strong>
          <button type="button" className="btn-close" onClick={onClose} aria-label="Fechar" />
        </div>
        <div className="card-body">
          <GuideContent />
        </div>
      </div>
    </>
  );
}

export default function App() {
  const [tokenError, setTokenError] = useState("");

  const [token, setToken] = useState<string | null>(null);
  const [inputToken, setInputToken] = useState("");

  const [assignment, setAssignment] = useState<AssignmentResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [saved, setSaved] = useState(false);

  const [speech, setSpeech] = useState("");
  const [offensive, setOffensive] = useState("");
  const [aspects, setAspects] = useState<string[]>([]);

  const [progress, setProgress] = useState<ProgressResponse | null>(null);
  const [showGuide, setShowGuide] = useState(false);

  const fetchNext = async () => {
    if (!token) return;

    setLoading(true);
    setTokenError("");

    try {
      const res = await fetch(`${API_BASE}/api/next?token=${token}`);

      if (res.status === 401 || res.status === 403) {
        clearToken();
        setToken(null);
        setInputToken("");
        setTokenError("Token inválido. Verifique e tente novamente.");
        setLoading(false);
        return;
      }

      const data: AssignmentResponse = await res.json();

      if (data.done) {
        setAssignment(null);
      } else if (!data.comment) {
        setAssignment(null);
        setTokenError("Erro ao carregar comentário. Avise o pesquisador.");
        setToken(null);
      } else {
        setAssignment(data);
      }

      setSpeech("");
      setOffensive("");
      setAspects([]);
    } catch (err) {
      console.error(err);
      setTokenError("Erro de conexão com o servidor.");
      setToken(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const cached = loadToken();
    if (cached) {
      setToken(cached);
    }
  }, []);

  useEffect(() => {
    if (token) {
      fetchNext();
      fetchProgress();
    }
  }, [token]);

  const submit = async () => {
    if (!assignment || !token) return;

    if (!speech || !offensive) {
      alert("Preencha todas as classes obrigatórias");
      return;
    }

    await fetch(`${API_BASE}/api/submit`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        token,
        assignment_id: assignment.assignment_id,
        comment_id: assignment.comment_id,
        speech_class: speech,
        offensive_class: offensive,
        aspects,
      }),
    });

    setSaved(true);

    setTimeout(() => {
      setSaved(false);
      fetchProgress();
      fetchNext();
    }, 800);
  };

  const fetchProgress = async () => {
    if (!token) return;

    const res = await fetch(`${API_BASE}/api/progress?token=${token}`);
    const data: ProgressResponse = await res.json();
    setProgress(data);
  };

  const saveToken = (token: string) => {
    const expiresAt = Date.now() + TOKEN_TTL_MINUTES * 60 * 1000;
    sessionStorage.setItem(TOKEN_CACHE_KEY, token);
    sessionStorage.setItem(TOKEN_EXP_KEY, expiresAt.toString());
  };

  const loadToken = (): string | null => {
    const token = sessionStorage.getItem(TOKEN_CACHE_KEY);
    const exp = sessionStorage.getItem(TOKEN_EXP_KEY);

    if (!token || !exp) return null;

    if (Date.now() > Number(exp)) {
      sessionStorage.removeItem(TOKEN_CACHE_KEY);
      sessionStorage.removeItem(TOKEN_EXP_KEY);
      return null;
    }

    return token;
  };

  const clearToken = () => {
    sessionStorage.removeItem(TOKEN_CACHE_KEY);
    sessionStorage.removeItem(TOKEN_EXP_KEY);
  };

  if (!token) {
    return (
      <div className="container mt-5 mb-5" style={{ maxWidth: 520 }}>
        <GuideModal open={showGuide} onClose={() => setShowGuide(false)} />

        <div className="d-flex justify-content-between align-items-center mb-3">
          <h4 className="mb-0">Sistema de Anotação</h4>
          <GuideButton onClick={() => setShowGuide(true)} />
        </div>

        <p className="text-muted">
          Este sistema tem como objetivo coletar anotações sobre comentários do YouTube
          relacionados à comunidade trans.
        </p>

        <div className="alert alert-light border" style={{ fontSize: "0.95rem" }}>
          Leia o guia antes de iniciar. Depois, ele continuará disponível durante toda a
          anotação pelo botão <strong>“Guia de anotação”</strong>.
        </div>

        <p className="text-muted">
          Insira o token fornecido pelo pesquisador para iniciar as anotações.
        </p>

        <input
          className="form-control mb-3"
          placeholder="Digite seu token"
          value={inputToken}
          onChange={(e) => setInputToken(e.target.value)}
        />

        <button
          className="btn btn-primary w-100"
          disabled={!inputToken}
          onClick={() => {
            const t = inputToken.trim();
            saveToken(t);
            setToken(t);
          }}
        >
          Iniciar anotações
        </button>

        {tokenError && (
          <div className="alert alert-danger mt-3">{tokenError}</div>
        )}
      </div>
    );
  }

  if (loading) {
    return <div className="container mt-5">Carregando…</div>;
  }

  if (!assignment) {
    return (
      <div className="container mt-5">
        <GuideModal open={showGuide} onClose={() => setShowGuide(false)} />

        <div className="d-flex justify-content-end mb-3">
          <GuideButton onClick={() => setShowGuide(true)} />
        </div>

        <div className="alert alert-success mt-3">
          🎉 Todas as anotações concluídas. Obrigado pela participação!
        </div>
      </div>
    );
  }

  return (
    <div className="container mt-4 mb-5" style={{ maxWidth: 900 }}>
      <GuideModal open={showGuide} onClose={() => setShowGuide(false)} />

      <div className="d-flex justify-content-between align-items-start mb-3 gap-3">
        <div className="flex-grow-1">
          {progress && (
            <div>
              <div className="d-flex justify-content-between">
                <small>
                  Progresso: {progress.done} / {progress.total}
                </small>
                <small>{Math.round((progress.done / progress.total) * 100)}%</small>
              </div>

              <div className="progress">
                <div
                  className="progress-bar"
                  role="progressbar"
                  style={{
                    width: `${(progress.done / progress.total) * 100}%`,
                  }}
                  aria-valuenow={progress.done}
                  aria-valuemin={0}
                  aria-valuemax={progress.total}
                />
              </div>
            </div>
          )}
        </div>

        <div style={{ flexShrink: 0 }}>
          <GuideButton onClick={() => setShowGuide(true)} />
        </div>
      </div>

      <h6 className="text-muted mb-1">Título do vídeo</h6>
      <h5 className="mb-3">{assignment.video_title}</h5>

      <h6 className="text-muted mb-1">Comentário</h6>
      <div className="card mb-3">
        <div className="card-body">
          <p className="mb-0">{assignment.comment}</p>
        </div>
      </div>

      <div className="mb-3">
        <label
          className="form-label"
          title="Classifique a atitude principal do comentário em relação a pessoas trans."
        >
          Discurso
        </label>
        <select
          className="form-select"
          value={speech}
          onChange={(e) => setSpeech(e.target.value)}
        >
          <option value="">Selecione</option>
          <option value="SUPPORTIVE">Apoio</option>
          <option value="NEUTRAL">Neutro</option>
          <option value="HATEFUL">Ódio / Intolerância</option>
        </select>
      </div>

      <div className="mb-3">
        <label
          className="form-label"
          title="Marque como ofensivo quando houver xingamento, humilhação ou agressividade verbal explícita."
        >
          É ofensivo? (ex.: contém palavras explícitas ou expressões de humilhação)
        </label>
        <select
          className="form-select"
          value={offensive}
          onChange={(e) => setOffensive(e.target.value)}
        >
          <option value="">Selecione</option>
          <option value="NON_OFFENSIVE">Não ofensivo</option>
          <option value="OFFENSIVE">Ofensivo</option>
        </select>
      </div>

      <div className="mb-3">
        <label
          className="form-label"
          title="Selecione uma ou mais vertentes que apareçam no foco do comentário."
        >
          Vertentes
        </label>
        {Object.entries(ASPECT_LABELS).map(([value, label]) => (
          <div className="form-check" key={value}>
            <input
              type="checkbox"
              className="form-check-input"
              checked={aspects.includes(value)}
              onChange={() =>
                setAspects((prev) =>
                  prev.includes(value)
                    ? prev.filter((x) => x !== value)
                    : [...prev, value],
                )
              }
            />
            <label className="form-check-label">{label}</label>
          </div>
        ))}
      </div>

      <button className="btn btn-primary" onClick={submit} disabled={saved}>
        Salvar e próximo
      </button>

      {saved && <div className="alert alert-success mt-3">Salvo ✓</div>}
    </div>
  );
}

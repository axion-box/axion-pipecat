import { useCallback, useEffect, useRef, useState } from 'react';
import type { FormEvent } from 'react';

import type { RTVIEventHandler, TranscriptData } from '@pipecat-ai/client-js';
import { RTVIEvent } from '@pipecat-ai/client-js';
import { usePipecatClientTransportState, useRTVIClientEvent } from '@pipecat-ai/client-react';
import type { PipecatBaseChildProps } from '@pipecat-ai/voice-ui-kit';
import { ConnectButton, EventsPanel, UserAudioControl } from '@pipecat-ai/voice-ui-kit';

interface AppProps extends PipecatBaseChildProps {}

type Phase = 'idle' | 'listening' | 'processing' | 'speaking';
type TurnRole = 'user' | 'assistant';
type ChatTurn = {
  id: string;
  role: TurnRole;
  text: string;
  createdAt: string;
  updatedAt: string;
};

const formatTimestamp = (value?: string) => {
  if (!value) {
    return '';
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return '';
  }

  return date.toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
  });
};

const roleLabel: Record<TurnRole, string> = {
  user: 'You',
  assistant: 'Assistant',
};

const makeTurn = (role: TurnRole, text = ''): ChatTurn => {
  const now = new Date().toISOString();
  return {
    id: `${role}-${now}-${Math.random().toString(36).slice(2, 8)}`,
    role,
    text,
    createdAt: now,
    updatedAt: now,
  };
};

const getPhaseLabel = (phase: Phase, transportState: string) => {
  if (transportState === 'error') {
    return 'Connection error';
  }

  switch (phase) {
    case 'listening':
      return 'Listening';
    case 'processing':
      return 'Thinking';
    case 'speaking':
      return 'Speaking';
    default:
      if (transportState === 'ready' || transportState === 'connected') {
        return 'Ready';
      }
      if (transportState === 'connecting' || transportState === 'authenticating') {
        return 'Connecting';
      }
      return 'Disconnected';
  }
};

const getPhaseTone = (phase: Phase, transportState: string) => {
  if (transportState === 'error') {
    return 'danger';
  }

  switch (phase) {
    case 'listening':
      return 'listening';
    case 'processing':
      return 'processing';
    case 'speaking':
      return 'speaking';
    default:
      if (transportState === 'ready' || transportState === 'connected') {
        return 'ready';
      }
      return 'idle';
  }
};

export const App = ({ client, handleConnect, handleDisconnect }: AppProps) => {
  const transportState = usePipecatClientTransportState();
  const [draft, setDraft] = useState('');
  const [submitError, setSubmitError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [phase, setPhase] = useState<Phase>('idle');
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [liveUserTranscript, setLiveUserTranscript] = useState('');
  const [activeAssistantTurn, setActiveAssistantTurn] = useState<ChatTurn | null>(null);
  const [assistantTurnCount, setAssistantTurnCount] = useState(0);
  const conversationEndRef = useRef<HTMLDivElement | null>(null);
  const activeAssistantTurnRef = useRef<ChatTurn | null>(null);
  const assistantSpeakingRef = useRef(false);
  const assistantLlmDoneRef = useRef(false);

  const handleConnectClick = async () => {
    await client?.initDevices();
    await handleConnect?.();
  };

  useEffect(() => {
    conversationEndRef.current?.scrollIntoView({ block: 'end' });
  }, [turns, liveUserTranscript, activeAssistantTurn]);

  const syncActiveAssistantTurn = useCallback((turn: ChatTurn | null) => {
    activeAssistantTurnRef.current = turn;
    setActiveAssistantTurn(turn);
  }, []);

  const appendUserTurn = useCallback((text: string) => {
    const trimmed = text.trim();
    if (!trimmed) {
      return;
    }

    setTurns((current) => [...current, makeTurn('user', trimmed)]);
  }, []);

  const ensureActiveAssistantTurn = useCallback(() => {
    const existing = activeAssistantTurnRef.current;
    if (existing) {
      return existing;
    }

    const created = makeTurn('assistant');
    syncActiveAssistantTurn(created);
    return created;
  }, [syncActiveAssistantTurn]);

  const appendAssistantDelta = useCallback(
    (delta: string) => {
      if (!delta) {
        return;
      }

      const current = ensureActiveAssistantTurn();
      syncActiveAssistantTurn({
        ...current,
        text: current.text + delta,
        updatedAt: new Date().toISOString(),
      });
    },
    [ensureActiveAssistantTurn, syncActiveAssistantTurn]
  );

  const finalizeActiveAssistantTurn = useCallback(() => {
    const current = activeAssistantTurnRef.current;
    if (!current) {
      assistantLlmDoneRef.current = false;
      assistantSpeakingRef.current = false;
      syncActiveAssistantTurn(null);
      return;
    }

    const trimmed = current.text.trim();
    if (trimmed) {
      const finalizedTurn = {
        ...current,
        text: trimmed,
        updatedAt: new Date().toISOString(),
      };
      setTurns((turnsCurrent) => [...turnsCurrent, finalizedTurn]);
      setAssistantTurnCount((count) => count + 1);
    }

    assistantLlmDoneRef.current = false;
    assistantSpeakingRef.current = false;
    syncActiveAssistantTurn(null);
  }, [syncActiveAssistantTurn]);

  useEffect(() => {
    if (transportState === 'ready' || transportState === 'connected') {
      if (phase === 'idle') {
        setPhase('listening');
      }
      return;
    }

    if (transportState === 'disconnected') {
      finalizeActiveAssistantTurn();
      setPhase('idle');
    }
  }, [finalizeActiveAssistantTurn, phase, transportState]);

  useRTVIClientEvent(
    RTVIEvent.UserStartedSpeaking,
    (() => {
      finalizeActiveAssistantTurn();
      setPhase('listening');
      setLiveUserTranscript('');
      assistantSpeakingRef.current = false;
    }) as RTVIEventHandler<RTVIEvent.UserStartedSpeaking>
  );

  useRTVIClientEvent(
    RTVIEvent.UserTranscript,
    ((data: TranscriptData) => {
      if (data.final) {
        appendUserTurn(data.text);
        setLiveUserTranscript('');
      } else {
        setLiveUserTranscript(data.text);
      }
    }) as RTVIEventHandler<RTVIEvent.UserTranscript>
  );

  useRTVIClientEvent(
    RTVIEvent.BotLlmStarted,
    (() => {
      finalizeActiveAssistantTurn();
      assistantLlmDoneRef.current = false;
      setPhase('processing');
    }) as RTVIEventHandler<RTVIEvent.BotLlmStarted>
  );

  useRTVIClientEvent(
    RTVIEvent.BotLlmText,
    ((data: { text: string }) => {
      appendAssistantDelta(data.text);
    }) as RTVIEventHandler<RTVIEvent.BotLlmText>
  );

  useRTVIClientEvent(
    RTVIEvent.BotLlmStopped,
    (() => {
      assistantLlmDoneRef.current = true;
      if (!assistantSpeakingRef.current) {
        finalizeActiveAssistantTurn();
        setPhase('listening');
      }
    }) as RTVIEventHandler<RTVIEvent.BotLlmStopped>
  );

  useRTVIClientEvent(
    RTVIEvent.BotStartedSpeaking,
    (() => {
      assistantSpeakingRef.current = true;
      setPhase('speaking');
    }) as RTVIEventHandler<RTVIEvent.BotStartedSpeaking>
  );

  useRTVIClientEvent(
    RTVIEvent.BotStoppedSpeaking,
    (() => {
      assistantSpeakingRef.current = false;
      if (assistantLlmDoneRef.current) {
        finalizeActiveAssistantTurn();
        setPhase('listening');
      } else if (activeAssistantTurnRef.current?.text.trim()) {
        setPhase('processing');
      } else {
        setPhase('listening');
      }
    }) as RTVIEventHandler<RTVIEvent.BotStoppedSpeaking>
  );

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    const text = draft.trim();
    if (!client || !text) {
      return;
    }

    setIsSubmitting(true);
    setSubmitError('');

    try {
      await client.sendText(text, {
        run_immediately: true,
        audio_response: true,
      });
      appendUserTurn(text);
      setDraft('');
    } catch (error) {
      setSubmitError(error instanceof Error ? error.message : 'Unable to send message.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const statusLabel = getPhaseLabel(phase, transportState);
  const statusTone = getPhaseTone(phase, transportState);
  const canSendText = transportState === 'ready' || transportState === 'connected';

  return (
    <div className="debugger-shell">
      <header className="debugger-header">
        <div className="debugger-brand">
          <div className="debugger-eyebrow">Axion Voice Debugger</div>
          <h1>Plain Voice Loop</h1>
          <p>
            Speak naturally, or type a message. The UI shows what the user said,
            when the model is thinking, and when TTS is actually speaking.
          </p>
        </div>

        <div className="debugger-controls">
          <div className={`status-pill status-pill-${statusTone}`}>{statusLabel}</div>
          <div className="control-row">
            <UserAudioControl size="lg" />
            <ConnectButton
              size="lg"
              onConnect={handleConnectClick}
              onDisconnect={handleDisconnect}
            />
          </div>
        </div>
      </header>

      <main className="debugger-main">
        <section className="conversation-card">
          <div className="conversation-meta">
            <div>
              <strong>{turns.length + (activeAssistantTurn ? 1 : 0)}</strong> messages
            </div>
            <div>
              Assistant turns: <strong>{assistantTurnCount}</strong>
            </div>
            <div>Transport: {transportState}</div>
          </div>

          <div className="conversation-scroll">
            {turns.length === 0 && !activeAssistantTurn ? (
              <div className="conversation-empty">
                Connect first, then speak. If the mic is blocked in the embedded browser,
                open the same URL in your system browser and allow microphone access.
              </div>
            ) : null}

            {turns.map((message) => (
              <article
                key={message.id}
                className={`message-bubble message-${message.role}`}
              >
                <div className="message-header">
                  <span>{roleLabel[message.role]}</span>
                  <span>{formatTimestamp(message.updatedAt ?? message.createdAt)}</span>
                </div>
                <div className="message-body">{message.text}</div>
              </article>
            ))}

            {liveUserTranscript ? (
              <article className="message-bubble message-user message-live">
                <div className="message-header">
                  <span>You</span>
                  <span>Live</span>
                </div>
                <div className="message-body">{liveUserTranscript}</div>
              </article>
            ) : null}

            {activeAssistantTurn ? (
              <article className="message-bubble message-assistant message-live">
                <div className="message-header">
                  <span>Assistant</span>
                  <span>{phase === 'speaking' ? 'Speaking' : 'Thinking'}</span>
                </div>
                <div className="message-body">
                  {activeAssistantTurn.text || 'Working on a reply...'}
                </div>
              </article>
            ) : null}

            <div ref={conversationEndRef} />
          </div>

          <form className="composer" onSubmit={handleSubmit}>
            <input
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              placeholder={
                canSendText
                  ? 'Type a message if you want to test text input.'
                  : 'Connect to send text or start voice capture.'
              }
              disabled={!canSendText || isSubmitting}
            />
            <button type="submit" disabled={!canSendText || isSubmitting || !draft.trim()}>
              {isSubmitting ? 'Sending...' : 'Send'}
            </button>
          </form>

          {submitError ? <div className="composer-error">{submitError}</div> : null}
        </section>

        <aside className="sidebar-card">
          <section className="sidebar-section">
            <h2>Session</h2>
            <ul className="sidebar-list">
              <li>Use the mic button to confirm the browser sees your microphone.</li>
              <li>Status changes to Thinking only after a user turn is committed.</li>
              <li>Assistant subtitles stream here before or while audio plays.</li>
            </ul>
          </section>

          <details className="debug-details">
            <summary>Debug event stream</summary>
            <div className="events-wrap">
              <EventsPanel />
            </div>
          </details>
        </aside>
      </main>
    </div>
  );
};

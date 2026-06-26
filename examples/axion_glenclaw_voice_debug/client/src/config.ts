import type { APIRequest } from '@pipecat-ai/client-js';

import type { TransportConnectionParams } from '@pipecat-ai/client-js';

export type TransportType = 'websocket';

export const AVAILABLE_TRANSPORTS: TransportType[] = ['websocket'];
export const DEFAULT_TRANSPORT: TransportType = 'websocket';

const botStartUrl =
  import.meta.env.VITE_BOT_START_URL || 'http://127.0.0.1:7860/start';

const normalizeLocalWsUrl = (rawUrl: string): string => {
  const url = new URL(rawUrl);
  const isLoopbackHost =
    url.hostname === '127.0.0.1' ||
    url.hostname === 'localhost' ||
    url.hostname === '::1';

  if (isLoopbackHost && url.protocol === 'wss:') {
    url.protocol = 'ws:';
  }

  return url.toString();
};

const websocketResponseTransformer = (
  response: TransportConnectionParams
): TransportConnectionParams => {
  const { wsUrl, token } = response as { wsUrl: string; token?: string };
  return {
    wsUrl: token
      ? `${normalizeLocalWsUrl(wsUrl)}?token=${encodeURIComponent(token)}`
      : normalizeLocalWsUrl(wsUrl),
  };
};

export const TRANSPORT_PROPS: Record<
  TransportType,
  {
    startBotParams: APIRequest;
    startBotResponseTransformer: (
      response: TransportConnectionParams
    ) => TransportConnectionParams;
  }
> = {
  websocket: {
    startBotParams: {
      endpoint: botStartUrl,
      requestData: {
        transport: 'websocket',
      },
    },
    startBotResponseTransformer: websocketResponseTransformer,
  },
};

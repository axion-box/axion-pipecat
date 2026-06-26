import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';

import type { PipecatBaseChildProps } from '@pipecat-ai/voice-ui-kit';
import {
  ErrorCard,
  FullScreenContainer,
  PipecatAppBase,
  SpinLoader,
  ThemeProvider,
} from '@pipecat-ai/voice-ui-kit';

import { App } from './components/App';
import { DEFAULT_TRANSPORT, TRANSPORT_PROPS } from './config';
import './index.css';

const Main = () => {
  const transportProps = TRANSPORT_PROPS[DEFAULT_TRANSPORT];

  return (
    <ThemeProvider defaultTheme="terminal" disableStorage>
      <FullScreenContainer>
        <PipecatAppBase
          {...transportProps}
          transportType={DEFAULT_TRANSPORT}>
          {({
            client,
            handleConnect,
            handleDisconnect,
            error,
          }: PipecatBaseChildProps) =>
            !client ? (
              <SpinLoader />
            ) : error ? (
              <ErrorCard>{error}</ErrorCard>
            ) : (
              <App
                client={client}
                handleConnect={handleConnect}
                handleDisconnect={handleDisconnect}
              />
            )
          }
        </PipecatAppBase>
      </FullScreenContainer>
    </ThemeProvider>
  );
};

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Main />
  </StrictMode>
);

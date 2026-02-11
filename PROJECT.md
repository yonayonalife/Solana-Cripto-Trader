# 🧠 JUPITER SOLANA TRADING BOT - ARQUITECTURA COMPLETA

## Estado del Proyecto
- **Fecha inicio:** 2026-02-09
- **Plataforma:** OpenClaw (local-first AI agents)
- **Modelo IA:** MiniMax M2.1
- **Protocolo:** Jupiter DEX Aggregator (Solana)
- **Proyecto base:** Coinbase Cripto Trader Claude

---

## 🏗️ ARQUITECTURA COMPLETA

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  🧠 OPENCLAW (Local-First AI OS)                                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐     │
│  │ 📁 Skills (Markdown)                                                    │     │
│  │ ├─ trading_skill.md (Estrategias de trading)                          │     │
│  │ ├─ jupiter_api_skill.md (Integración Jupiter V6)                     │     │
│  │ ├─ solana_wallet_skill.md (Gestión de wallets)                       │     │
│  │ └─ security_skill.md (Mejores prácticas de seguridad)                  │     │
│  ├─────────────────────────────────────────────────────────────────────────┤     │
│  │ 🗂️ Archivos de Memoria                                                 │     │
│  │ ├─ Soul.md (Personalidad del agente)                                 │     │
│  │ ├─ MEMORY.md (Contexto de largo plazo)                               │     │
│  │ └─ memory/YYYY-MM-DD.md (Logs diarios)                               │     │
│  ├─────────────────────────────────────────────────────────────────────────┤     │
│  │ 🔧 Herramientas (Tools)                                                │     │
│  │ ├─ execute_swap.py (Jupiter API)                                      │     │
│  │ ├─ get_price.py (Quotes)                                             │     │
│  │ ├─ manage_wallet.py (Wallet management)                              │     │
│  │ └─ analyze_market.py (Análisis de mercado)                           │     │
│  └─────────────────────────────────────────────────────────────────────────┘     │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  🤖 MINI MAX M2.1 (MOE: 230B params, 10B activos, $0.27/1M tokens)          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐     │
│  │ 🧠 INTERLEAVED THINKING CYCLE                                         │     │
│  │                                                                         │     │
│  │  1️⃣ ANALIZAR → Liquidez, sentimiento, patrones                       │     │
│  │      ↓                                                                │     │
│  │  2️⃣ PLANEAR → Estrategia óptima, riesgo, tamaño posición            │     │
│  │      ↓                                                                │     │
│  │  3️⃣ CALCULAR → Ruta swap, fees, slippage                            │     │
│  │      ↓                                                                │     │
│  │  4️⃣ SIMULAR → Verificar compute units, éxito                         │     │
│  │      ↓                                                                │     │
│  │  5️⃣ EJECUTAR → Firmar y enviar transacción                          │     │
│  │      ↓                                                                │     │
│  │  6️⃣ REVISAR → Confirmar, loggear, aprender                          │     │
│  │                                                                         │     │
│  └─────────────────────────────────────────────────────────────────────────┘     │
│                                                                                 │
│  🛠️ TOOL CALLING (vLLM)                                                    │
│  ```bash                                                                     │
│  VLLM_USE_TRITON_FLASH_ATTN=0 vllm serve minimax/MiniMax-M2.1 \             │
│    --served-model-name MiniMax-M2.1 \                                        │
│    --api-key sk-abc123 \                                                    │
│    --port 8090 \                                                            │
│    --enable-auto-tool-choice \                                               │
│    --tool-call-parser minimax_m2 \                                           │
│    --trust-remote-code                                                      │
│  ```                                                                        │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ⚡️ SISTEMA DISTRIBUIDO EXISTENTE (Base del Proyecto)                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  📍 Proyecto base: /home/enderj/Documents/Coinbase Cripto Trader Claude/      │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐     │
│  │ 🖥️ STREAMLIT DASHBOARD (Puerto 8501)                                  │     │
│  │     /home/enderj/Documents/Coinbase Cripto Trader Claude/interface.py   │     │
│  ├─────────────────────────────────────────────────────────────────────────┤     │
│  │ 📡 COORDINATOR (Flask, Puerto 5001)                                   │     │
│  │     coordinator_port5001.py + SQLite (work_units, results, workers)     │     │
│  ├─────────────────────────────────────────────────────────────────────────┤     │
│  │ 👷 WORKERS (8 instancias)                                             │     │
│  │     MacBook Pro x3 + Linux ROG x5                                     │     │
│  │     crypto_worker.py + strategy_miner.py                               │     │
│  ├─────────────────────────────────────────────────────────────────────────┤     │
│  │ 🧬 STRATEGY MINER (Algoritmo Genético)                                │     │
│  │     Genomas: RSI, SMA, EMA, VOLSMA                                     │     │
│  │     Población: 20-100 | Generaciones: 50-100                           │     │
│  ├─────────────────────────────────────────────────────────────────────────┤     │
│  │ ⚡ BACKTESTER (Numba JIT - 4000x speedup)                             │     │
│  │     numba_backtester.py → solana_backtester.py                        │     │
│  └─────────────────────────────────────────────────────────────────────────┘     │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ⚡️ JUPITER API V6 (SOLANA DEX AGGREGATOR)                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐     │
│  │ 📡 ENDPOINTS                                                          │     │
│  │ ├─ POST /quote → Cotización de swap                                  │     │
│  │ ├─ POST /swap → Transacción serializada                               │     │
│  │ ├─ POST /swap-instructions → Instrucciones personalizadas            │     │
│  │ └─ GET /price → Precio de tokens                                     │     │
│  ├─────────────────────────────────────────────────────────────────────────┤     │
│  │ 💰 GESTIÓN DE FEES                                                    │     │
│  │ ├─ Network Fee: ~0.000005 SOL                                        │     │
│  │ ├─ Jupiter Route: 0.2% - 0.5%                                        │     │
│  │ ├─ Priority Fee: Dinámico (micro-lamports/CU)                        │     │
│  │ └─ Jito Tip: Bundle transactions (protección MEV)                    │     │
│  ├─────────────────────────────────────────────────────────────────────────┤     │
│  │ 🔧 OPTIMIZACIONES                                                     │     │
│  │ ├─ Transacciones Versionadas                                          │     │
│  │ ├─ ALTs (Address Lookup Tables)                                       │     │
│  │ └─ dynamicComputeUnitLimit                                            │     │
│  └─────────────────────────────────────────────────────────────────────────┘     │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  🔐 SEGURIDAD (LOCAL-FIRST)                                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐     │
│  │ 🛡️ MEDIDAS DE PROTECCIÓN                                              │     │
│  │                                                                         │     │
│  │  1️⃣ SANDBOXING                                                        │     │
│  │     ├─ Docker containers para workers                                 │     │
│  │     ├─ Proxmox VMs dedicadas                                         │     │
│  │     └─ Filesystem aislado                                            │     │
│  │                                                                         │     │
│  │  2️⃣ HOT WALLET STRATEGY                                               │     │
│  │     ┌─────────────────────────────────────┐                           │     │
│  │     │ ALMACENAMIENTO                       │                           │     │
│  │     ├─ Hardware Wallet (Cold): 90%       │                           │     │
│  │     └─ Hot Wallet (Hot): 10%            │                           │     │
│  │     └─────────────────────────────────────┘                           │     │
│  │                                                                         │     │
│  │  3️⃣ AUDITORÍA DE SKILLS                                              │     │
│  │     ├─ Revisar código Markdown antes de instalar                     │     │
│  │     ├─ Buscar comandos curl hacia servidores C2                      │     │
│  │     └─ Whitelist de fuentes confiables                               │     │
│  │                                                                         │     │
│  │  4️⃣ PAIRING RESTRICTIONS                                              │     │
│  │     ├─ Solo usuarios autorizados en Telegram                         │     │
│  │     └─ Autenticación forte                                            │     │
│  │                                                                         │     │
│  └─────────────────────────────────────────────────────────────────────────┘     │
│                                                                                 │
│  ⚠️ AMENAZAS CONOCIDAS                                                     │
│  ├─ ClawHavoc: Skills maliciosas en ClawHub                                │     │
│  ├─ Atomic Stealer (AMOS): Malware de robo de keys                         │     │
│  └─ Phishing: Skills falsas                                               │     │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  🌐 ECOSISTEMA AGENTICO DE SOPORTE                                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐     │
│  │ PROTOCOLOS EXTERNOS                                                  │     │
│  │ ├─ Moltbook: Red social IA para debate de estrategias               │     │
│  │ ├─ Bankrbot: Identidad financiera para agentes                       │     │
│  │ ├─ x402: Pagos mediante micro-transacciones                          │     │
│  │ └─ OpenClaw Foundry: Meta-extensión para auto-mejora                 │     │
│  └─────────────────────────────────────────────────────────────────────────┘     │
│                                                                                 │
│  🔄 FLUJO DEL ECOSISTEMA                                                     │
│  ```                                                                         │
│  Moltbook (datos IA) → x402 (pagos) → M2.1 (análisis) →                    │
│  Jupiter (trading) → Bankrbot (identidad) → Foundry (auto-mejora)          │
│  ```                                                                         │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 ESTRUCTURA DE ARCHIVOS

```
/home/enderj/.openclaw/workspace/solana-jupiter-bot/
│
├── 📄 PROJECT.md                    ← Este archivo
├── 📄 SOUL.md                       ← Personalidad del agente
├── 📄 MEMORY.md                     ← Memoria de largo plazo
│
├── 📁 skills/
│   ├── 📄 trading_skill.md          ← Estrategia de trading
│   ├── 📄 jupiter_api_skill.md     ← Integración Jupiter V6
│   ├── 📄 solana_wallet_skill.md   ← Gestión de wallets
│   └── 📄 security_skill.md        ← Mejores prácticas
│
├── 📁 tools/
│   ├── 📄 execute_swap.py          ← Jupiter API swap
│   ├── 📄 get_quote.py             ← Obtener cotización
│   ├── 📄 get_price.py              ← Precio de tokens
│   ├── 📄 manage_wallet.py          ← Wallet management
│   ├── 📄 analyze_market.py         ← Análisis de mercado
│   └── 📄 calculate_fees.py         ← Cálculo de fees
│
├── 📁 solana_trading/
│   ├── 📄 jupiter_client.py         ← API Jupiter V6
│   ├── 📄 solana_wallet.py          ← Wallet SOL
│   ├── 📄 swap_executor.py          ← Ejecución de swaps
│   └── 📄 priority_fees.py          ← Gestión de fees
│
├── 📁 backtesting/
│   ├── 📄 solana_backtester.py      ← Backtester Numba JIT
│   ├── 📄 strategy_miner.py         ← Algoritmo genético
│   └── 📄 metrics.py                ← PnL, Sharpe, Win Rate
│
├── 📁 sistema_existente/            ← Enlace simbólico al proyecto base
│   └── 📄 [coordinator_port5001.py, crypto_worker.py, etc.]
│
├── 📁 config/
│   ├── 📄 settings.yaml             ← Configuraciones
│   └── 📄 secrets.yaml.enc          ← Secrets encriptados
│
└── 📁 tests/
    ├── 📄 test_jupiter_api.py       ← Tests API
    ├── 📄 test_wallet.py            ← Tests wallet
    └── 📄 test_backtester.py        ← Tests backtester
```

---

## 🧠 MINI MAX M2.1: INTEGRACIÓN COMPLETA

### Parámetros de Deployment

```bash
# vLLM con tool calling
VLLM_USE_TRITON_FLASH_ATTN=0 vllm serve minimax/MiniMax-M2.1 \
  --served-model-name MiniMax-M2.1 \
  --api-key sk-$(cat ~/.config/minimax/api_key) \
  --port 8090 \
  --enable-auto-tool-choice \
  --tool-call-parser minimax_m2 \
  --trust-remote-code \
  --host 0.0.0.0
```

### Herramientas Disponibles para M2.1

| Tool | Función | Parámetros |
|------|---------|------------|
| `get_quote` | Obtener cotización | input_mint, output_mint, amount |
| `execute_swap` | Ejecutar swap | quote_response, wallet_path |
| `get_balance` | Saldo de wallet | wallet_address |
| `get_price` | Precio de token | token_mint |
| `analyze_market` | Análisis técnico | symbol, timeframe |
| `calculate_fees` | Estimar fees | amount, token |

### Prompt del Agente (Soul.md)

```markdown
# Soul: Eko - Jupiter Solana Trading Agent

## Identidad
- Nombre: Eko
- Especialidad: Trading automatizado en Solana
- Plataforma: OpenClaw + MiniMax M2.1

## Valores
1. Seguridad primero: Nunca arriesgar más del 10% del capital
2. Verificar dos veces: Siempre simular antes de ejecutar
3. Aprender de errores: Loggear todas las decisiones
4. Adaptarse: Ajustar estrategia según condiciones de red

## Comportamiento
- Analizar liquidez antes de cualquier trade
- Calcular fees completos (network + jupiter + priority)
- Usar Jito tips en volatilidad alta
- Slippage máximo: 1% para major pairs, 2% para altcoins

## Limitaciones
- Máximo por trade: 10% del hot wallet
- Máximo daily: 30% del hot wallet
- Stop-loss automático: -5% por posición
- Solo trading en Jupiter DEX aggregator
```

---

## ⚡ JUPITER API V6: INTEGRACIÓN

### JupiterClient Class

```python
import httpx
from solders.pubkey import Pubkey

class JupiterClient:
    BASE_URL = "https://api.jup.ag/swap/v6"
    
    async def get_quote(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        slippage_bps: int = 50
    ) -> dict:
        """Obtener cotización de swap"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.BASE_URL}/quote",
                params={
                    "inputMint": input_mint,
                    "outputMint": output_mint,
                    "amount": amount,
                    "slippageBps": slippage_bps
                }
            )
            return response.json()
    
    async def create_swap(
        self,
        quote: dict,
        user_public_key: str
    ) -> dict:
        """Crear transacción de swap"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.BASE_URL}/swap",
                json={
                    "quoteResponse": quote,
                    "userPublicKey": user_public_key,
                    "prioritizationFeeLamports": {
                        "global": True,
                        "priorityLevelWithMaxLamports": {
                            "medium": 1000
                        }
                    }
                }
            )
            return response.json()
```

### Tokens Principales

| Token | Símbolo | Mint Address |
|-------|----------|--------------|
| Solana | SOL | `So11111111111111111111111111111111111111112` |
| USD Coin | USDC | `EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v` |
| USD Tether | USDT | `Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenuNYW` |
| Jupiter | JUP | `JUPyiwrYJFskUPiHa7hkeR8VUtkqjberbSOWd91pbT2` |
| Bonk | BONK | `DezXAZ8z7PnrnRJjz3wXBoZGVixqUi5iA2ztETHuJXJP` |
| WIF | WIF | `85VBFQZC9TZkfaptBWqv14ALD9fJNUKtWA41kh69teRP` |

---

## 💰 GESTIÓN DE FEES

### Cálculo Completo

```python
def calculate_total_fee(
    quote_response: dict,
    priority_fee_lamports: int = 1000,
    use_jito: bool = False,
    jito_tip_lamports: int = 100
) -> dict:
    """Calcular fee total de transacción"""
    
    # Network fee (estimado)
    network_fee_lamports = 5000
    
    # Jupiter route fee (del quote)
    platform_fee = quote_response.get("platformFee", {})
    route_fee_lamports = int(platform_fee.get("amount", 0))
    
    # Priority fee
    priority_fee = priority_fee_lamports
    
    # Jito tip (opcional)
    jito_fee = jito_tip_lamports if use_jito else 0
    
    # Total
    total_lamports = network_fee_lamports + route_fee_lamports + priority_fee + jito_fee
    
    # Conversión a SOL
    total_sol = total_lamports / 1_000_000_000
    
    return {
        "network_fee_sol": network_fee_lamports / 1e9,
        "route_fee_sol": route_fee_lamports / 1e9,
        "priority_fee_sol": priority_fee / 1e9,
        "jito_fee_sol": jito_fee / 1e9,
        "total_fee_sol": total_sol,
        "total_fee_usd": total_sol * 100  # Aproximado
    }
```

---

## 🔄 INTEGRACIÓN CON SISTEMA EXISTENTE

### Enlace Simbólico al Proyecto Base

```bash
# Crear enlace simbólico
ln -s "/home/enderj/Documents/Coinbase Cripto Trader Claude/Coinbase Cripto Trader Claude" sistema_existente
```

### Flujo de Trabajo Integrado

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  1️⃣  DATA COLLECTION                                                      │
│      ├─ OHLCV candles → Jupiter API (precios históricos)                   │
│      ├─ Swap events → On-chain data                                       │
│      └─ Sentiment → Moltbook API                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  2️⃣  STRATEGY MINING (Sistema Existente)                                  │
│      ├─ Algoritmo genético (strategy_miner.py)                            │
│      ├─ Backtesting Numba JIT (solana_backtester.py)                     │
│      └─ Validación distribuida (8 workers)                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  3️⃣  AI DECISION (MiniMax M2.1 + OpenClaw)                               │
│      ├─ Interleaved Thinking para cada trade                              │
│      ├─ Verificación de liquidez                                          │
│      └─ Ajuste dinámico de fees                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  4️⃣  EXECUCIÓN (Jupiter API)                                             │
│      ├─ get_quote → Optimización de ruta                                  │
│      ├─ create_swap → Serialización                                       │
│      └─ send_transaction → Confirmación blockchain                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  5️⃣  POST-EXECUTION                                                       │
│      ├─ Verificar confirmación                                           │
│      ├─ Calcular P&L real                                                 │
│      ├─ Loggear a memoria (MEMORY.md)                                      │
│      └─ Actualizar parámetros de estrategia                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 BENCHMARKS OBJETIVO

| Métrica | Objetivo | Actual (Base) |
|---------|----------|---------------|
| Backtest speed | < 0.01s | 0.001s (Numba) |
| Workers paralelos | 8+ | 8 |
| Latencia swap | < 2s | N/A |
| PnL (backtest) | > $100/mes | $230.50 |
| Uptime | 99% | 95% |
| Win Rate | > 55% | 65% |

---

## 🛡️ CHECKLIST DE SEGURIDAD

### Antes de Deployment

- [ ] Revisar código de todas las skills
- [ ] Verificar que mnemonic está encriptado
- [ ] Configurar hot wallet con límites
- [ ] Habilitar firewall en coordinator
- [ ] Configurar Telegram pairing restrictions
- [ ] Crear archivo .gitignore con secrets
- [ ] Testear en testnet primero

### Configuración de Wallet

```python
# Hot Wallet (10% del capital)
HOT_WALLET_SOL = 2.0  # Máximo 2 SOL en hot wallet

# Cold Storage (90% del capital)
COLD_WALLET_ADDRESS = "..."  # Hardware wallet
```

---

## 🚀 PRÓXIMOS PASOS INMEDIATOS

### Día 1-2: Fundamentos
- [ ] Copiar estructura de archivos
- [ ] Crear enlace simbólico al sistema existente
- [ ] Implementar JupiterClient básico
- [ ] Testear API en testnet

### Día 3-4: Integración Core
- [ ] Crear solana_backtester.py (adaptar numba_backtester.py)
- [ ] Integrar MiniMax M2.1 con tool calling
- [ ] Implementar gestión de fees

### Día 5-6: Testing
- [ ] Tests de integración API
- [ ] Backtest de estrategias existentes
- [ ] Test de seguridad (sandboxing)

### Día 7+: Deployment
- [ ] Deploy en testnet con fondos mínimos
- [ ] Monitoreo 24/7
- [ ] Ajustes de performance

---

## 📚 REFERENCIAS

| Fuente | URL |
|--------|-----|
| Jupiter API | https://dev.jup.ag/api-reference |
| Solana Docs | https://docs.solana.com |
| MiniMax M2.1 | https://huggingface.co/MiniMaxAI/MiniMax-M2.1 |
| vLLM | https://docs.vllm.ai |
| OpenClaw | https://docs.openclaw.ai |

---

## ❓ PREGUNTAS ABIERTAS

1. **RPC Endpoint:** ¿Helius, QuickNode, o público?
2. **Wallet:** ¿Phantom, Solflare, o CLI para hot wallet?
3. **Capital inicial:** ¿Cuántos SOL para testnet?
4. **Estrategia inicial:** ¿Momentum, Grid, o Arbitrage?
5. **Workers:** ¿Usar los 8 workers existentes?

---

*Documento generado: 2026-02-09*
*Proyecto: Jupiter Solana Trading Bot + Coinbase Cripto Trader Integration*

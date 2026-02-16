# Skill: Jupiter API V6 Integration

## Descripción
Esta skill proporciona las instrucciones y contexto para interactuar con la API V6 de Jupiter DEX Aggregator en Solana.

## Endpoints Disponibles

### Quote Endpoint
```
GET https://api.jup.ag/swap/v6/quote
```

**Parámetros:**
| Parámetro | Tipo | Requerido | Descripción |
|-----------|------|-----------|-------------|
| `inputMint` | string | ✅ | Token de entrada (mint address) |
| `outputMint` | string | ✅ | Token de salida (mint address) |
| `amount` | integer | ✅ | Amount en lamports (no decimals) |
| `slippageBps` | integer | ❌ | Slippage en basis points (default: 50) |
| `swapMode` | string | ❌ | `ExactIn` o `ExactOut` (default: ExactIn) |
| `platformFeeBps` | integer | ❌ | Fee de plataforma (default: 0) |

**Ejemplo de Request:**
```
GET https://api.jup.ag/swap/v6/quote?inputMint=So11111111111111111111111111111111111111112&outputMint=EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v&amount=1000000000&slippageBps=50
```

**Ejemplo de Response:**
```json
{
  "inputMint": "So11111111111111111111111111111111111111112",
  "inAmount": "1000000000",
  "outputMint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
  "outAmount": "17650000",
  "otherAmountThreshold": "17527500",
  "swapMode": "ExactIn",
  "slippageBps": 50,
  "priceImpactPct": "0.0002",
  "platformFee": {
    "amount": "0",
    "feeBps": 0,
    "feeMint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
  },
  "routePlan": [
    {
      "swapInfo": {
        "ammKey": "HXpGFJGCEEFdV31tDmjDBaJMEB1fKLiAoKoWr3Fnonid",
        "label": "Meteora DLMM",
        "inputMint": "So11111111111111111111111111111111111111112",
        "outputMint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        "inAmount": "1000000000",
        "outAmount": "17650000",
        "feeAmount": "26475",
        "feeMint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
      },
      "percent": 100
    }
  ],
  "contextSlot": 324307186,
  "timeTaken": 0.012
}
```

### Swap Endpoint
```
POST https://api.jup.ag/swap/v6/swap
```

**Body:**
```json
{
  "quoteResponse": { ... },
  "userPublicKey": "WALLET_ADDRESS_HERE",
  "wrapUnwrapSOL": true,
  "prioritizationFeeLamports": {
    "global": true,
    "priorityLevelWithMaxLamports": {
      "medium": 1000
    }
  }
}
```

### Swap Instructions Endpoint
```
POST https://api.jup.ag/swap/v6/swap-instructions
```

**Útil para:** Transacciones personalizadas, integración con otros programas.

---

## 🪙 Tokens Soportados

### Major Pairs
| Token | Símbolo | Mint Address | Decimals |
|-------|----------|--------------|----------|
| Solana | SOL | `So11111111111111111111111111111111111111112` | 9 |
| USD Coin | USDC | `EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v` | 6 |
| USD Tether | USDT | `Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenuNYW` | 6 |

### Popular Altcoins
| Token | Mint Address |
|-------|--------------|
| JUP | `JUPyiwrYJFskUPiHa7hkeR8VUtkqjberbSOWd91pbT2` |
| BONK | `DezXAZ8z7PnrnRJjz3wXBoZGVixqUi5iA2ztETHuJXJP` |
| WIF | `85VBFQZC9TZkfaptBWqv14ALD9fJNUKtWA41kh69teRP` |
| POPCAT | `DLpS9S4K7y1ohTW2hx5qzY1N1Kc8Y5w9h8g5K2zXxXx` |
| JTO | `jtojtomepa8beP8AuQc6eXt5FriJwfcoAAaosrqv6NQ` |
| HNT | `HntvSCa5Jb16e7Ey5b9bdnTMRQf4U32um7SbewW5vNk` |

---

## 💰 Cálculo de Decimals

```python
def lamports_to_decimal(lamports: int, decimals: int) -> float:
    return lamports / (10 ** decimals)

def decimal_to_lamports(amount: float, decimals: int) -> int:
    return int(amount * (10 ** decimals))

# Ejemplos
lamports_to_decimal(1000000000, 9)  # 1.0 SOL
decimal_to_lamports(100, 6)         # 100000000 USDC
```

---

## 🔧 Gestión de Fees

### Priority Fees (Lamports)
```yaml
# Niveles de priority fee
economy: 0
low: 500
medium: 1000
high: 5000
very_high: 10000
max: 50000
```

### Jito Tips (Protección MEV)
```yaml
# Jito tips opcionales
disabled: 0
small: 500
medium: 1000
large: 5000
```

### Costo Total Estimado
```
Network Fee: ~0.000005 SOL (5000 lamports)
Jupiter Route Fee: 0.2% - 0.5% del swap
Priority Fee: Variable (0 - 0.01 SOL)
Jito Tip: Opcional (0 - 0.001 SOL)
```

---

## 📋 Flujo de Ejecución

### Paso 1: Obtener Quote
```python
quote = await jupiter.get_quote(
    input_mint=SOL_MINT,
    output_mint=USDC_MINT,
    amount=decimal_to_lamports(1.0, 9),  # 1 SOL
    slippage_bps=50
)
```

### Paso 2: Verificar Quote
```python
# Verificar slippage real
slippage_actual = 1 - (int(quote["outAmount"]) / int(quote["outAmount"]) if quote["otherAmountThreshold"] else 1)
if slippage_actual > MAX_SLIPPAGE:
    skip_trade()
```

### Paso 3: Crear Transacción
```python
swap_data = await jupiter.create_swap(
    quote_response=quote,
    user_public_key=wallet.pubkey(),
    prioritization_fee_lamports={"global": True, "priorityLevelWithMaxLamports": {"medium": 1000}}
)
```

### Paso 4: Firmar y Enviar
```python
transaction = Transaction.from_base58(swap_data["transaction"])
signed = wallet.sign(transaction)
result = await connection.send_raw_transaction(signed.serialize())
```

### Paso 5: Confirmar
```python
confirmation = await connection.confirm_transaction(result, commitment="confirmed")
```

---

## ⚠️ Errores Comunes

| Error | Causa | Solución |
|-------|-------|----------|
| `INSUFFICIENT_LIQUIDITY` | No hay liquidez | Reducir amount o cambiar ruta |
| `SLIPPAGE_EXCEEDED` | Slippage > limite | Usar menor amount o más liquidity |
| `INVALID_MINT` | Token no soportado | Verificar mint address |
| `ACCOUNT_NOT_FOUND` | Wallet sin tokens | Depositar primero |
| `TOKEN_DECIMALS_MISMATCH` | Decimals wrong | Verificar conversión |

---

## 🛡️ Mejores Prácticas

### Antes de Swap
1. ✅ Verificar saldo de wallet
2. ✅ Calcular fees totales
3. ✅ Confirmar slippage acceptable
4. ✅ Verificar mint addresses

### Durante Swap
1. ✅ Usar priority fee adecuado
2. ✅ Considerar Jito tip en volatilidad
3. ✅ Wrap/Unwrap SOL si es necesario
4. ✅ Timeout de 30 segundos

### Después de Swap
1. ✅ Confirmar transacción
2. ✅ Verificar balance actualizado
3. ✅ Loggear para auditoría
4. ✅ Calcular P&L

---

## 🔗 Recursos

- **Docs Oficiales:** https://dev.jup.ag/api-reference
- **Swagger UI:** https://api.jup.ag/docs
- **Ejemplos:** https://github.com/jup-ag

---

*Skill Version: 1.0*
*Last Updated: 2026-02-09*

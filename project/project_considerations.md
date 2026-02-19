## 🎯 **PREMISSAS FUNDAMENTAIS**

### **1. Qualidade dos Dados (CRÍTICO)**

**Premissas que precisamos validar:**
- ✅ Dados representam comportamento real (não apenas subset enviesado)
- ✅ Período de análise é suficiente para padrões sazonais (mín. 12 meses)
- ✅ Transações canceladas estão marcadas corretamente
- ✅ CustomerID está presente na maioria das transações (>70%)

**Red flags comuns neste dataset UCI:**
- ~25% das transações SEM CustomerID → decisão: excluir ou imputar?
- Quantidades negativas = devoluções/cancelamentos
- Preços unitários ≤ 0 ou extremamente altos (outliers ou erros?)
- Invoices duplicadas ou parcialmente canceladas

**Ação Senior:**
```python
# SEMPRE documentar decisões de limpeza
data_quality_report = {
    "missing_customer_id": "24.93% - DECISÃO: Excluir (impossível segmentar)",
    "negative_quantities": "8,905 linhas - DECISÃO: Marcar como returns",
    "outlier_prices": "327 produtos >£1000 - DECISÃO: Manter (luxo válido)"
}
# Justificar para stakeholders!
```

---

### **2. Definição de Métricas Alinhadas ao Negócio**

**CUIDADO:** Métricas técnicas ≠ métricas de negócio

**Erros comuns:**
- ❌ "Consegui Silhouette Score de 0.65!" → **E daí? Como isso impacta revenue?**
- ❌ "Modelo de churn com 92% accuracy" → **Mas qual o custo de falso negativo?**

**Abordagem Senior:**
```python
# Definir métricas de NEGÓCIO primeiro
business_metrics = {
    "Segmentation Success": "Variância inter-cluster de CLV > 40%",
    "Actionability": "Cada segmento tem ≥500 clientes (viável para campanha)",
    "Churn Model ROI": "Saving > 3x campaign cost (não só accuracy)",
    "Recommendation CTR": "Lift >15% vs. baseline random recommendations"
}
```

**Framework de validação:**
- Modelo técnico pode ser perfeito, mas se não gerar ação → falhou

---

### **3. Contexto de Negócio do Varejo UK**

**Premissas importantes:**
- Dataset é de **varejo B2B** (atacado de presentes/decoração)
- Muitos clientes são **revendedores**, não consumidores finais
- Comportamento B2B ≠ B2C (compras em lote, sazonalidade diferente)

**Impacto nas decisões:**
- RFM tradicional pode não funcionar bem (compras bulk são normais)
- "Churn" pode ser sazonal (loja fecha no inverno)
- Recomendações devem considerar mix de produtos para revenda

**Ação Senior:**
```python
# Adaptar RFM para contexto B2B
rfm_thresholds = {
    "recency": [30, 60, 120, 240],  # B2B tem ciclos mais longos
    "frequency": [2, 5, 10, 20],     # Menos frequente que B2C
    "monetary": [500, 2000, 5000, 15000]  # Tickets muito maiores
}
```

---

## ⚠️ **PONTOS CRÍTICOS DE ATENÇÃO**

### **1. Data Leakage (MORTAL para credibilidade)**

**Cenários perigosos:**

**Leakage temporal:**
```python
# ❌ ERRADO - usando dados do futuro
X = df[['recency', 'frequency', 'monetary', 'total_revenue']]
# total_revenue inclui compras futuras!

# ✅ CORRETO - split temporal rigoroso
train_cutoff = '2011-09-01'
test_cutoff = '2011-12-01'

train = df[df['InvoiceDate'] < train_cutoff]
test = df[(df['InvoiceDate'] >= train_cutoff) & 
          (df['InvoiceDate'] < test_cutoff)]
```

**Leakage de features:**
```python
# ❌ ERRADO - "churn" é o que queremos prever
features = ['days_since_last', 'avg_basket', 'is_churned']

# ✅ CORRETO - apenas info disponível ANTES do evento
features = ['days_since_last', 'avg_basket', 'trend_last_3months']
```

---

### **2. Overfitting em Segmentação**

**Problema:** Criar 47 micro-segmentos que não são acionáveis

**Princípio Senior:**
```python
# Regra de ouro
min_segment_size = max(
    500,  # Mínimo absoluto para campanha
    len(customers) * 0.05  # Pelo menos 5% da base
)

# Validação de acionabilidade
for segment in segments:
    if segment['size'] < min_segment_size:
        print(f"⚠️ Segmento '{segment['name']}' muito pequeno - FUNDIR")
    
    if segment['revenue_variance'] < 0.15:
        print(f"⚠️ Segmento '{segment['name']}' sem diferenciação - REVISAR")
```

**Teste prático:**
> "Se eu apresentar isso para o CMO, ele consegue criar 1 campanha específica para este segmento?"

Se a resposta for não → segmento inútil

---

### **3. Interpretabilidade vs. Performance**

**Dilema comum:**
```python
# Modelo complexo: XGBoost com 150 features
# - Accuracy: 94%
# - Explicabilidade: ❌ "é tipo magia negra"

# vs.

# Modelo simples: Logistic Regression com 8 features
# - Accuracy: 87%
# - Explicabilidade: ✅ "cada aumento de 1 em X aumenta churn em Y%"
```

**Decisão Senior:**
- Para **segmentação**: SEMPRE priorize interpretabilidade (K-Means > DBSCAN)
- Para **scoring**: Balance performance com explicações (SHAP values ajudam)
- Para **produção**: Simplicidade > complexidade (manutenção futura)

**Regra de ouro:**
> "Se não consigo explicar em 2 minutos para o time de marketing, modelo não vai ser usado"

---

### **4. Viés de Sobrevivência (Survivorship Bias)**

**Problema:** Dataset só tem clientes que COMPRARAM

**O que está faltando:**
- Clientes que abandonaram carrinho
- Visitantes que nunca compraram
- Clientes que churned ANTES do período de análise

**Impacto:**
```python
# ❌ Conclusão enviesada
"Nosso churn rate é apenas 15%!"
# Mas 15% de QUEM? Só de quem já comprou pelo menos 2x...

# ✅ Conclusão correta
"Entre clientes ativos em Jan/2010, 15% não compraram mais até Dez/2011"
# Deixar claro o denominador
```

**Ação Senior:**
- Documentar limitações explicitamente
- Criar "cohorts" claros (ex: "clientes adquiridos em Q1 2011")
- Nunca generalizar além do escopo dos dados

---

### **5. Correlação ≠ Causalidade (CRÍTICO para recomendações)**

**Exemplo perigoso:**
```python
# Análise mostra:
"Clientes que compram produto A têm 3x mais lifetime value"

# ❌ Recomendação ingênua
"Vamos empurrar produto A para todos!"

# Problema: 
# Produto A é caro → só clientes ricos compram → CLV alto é CAUSA, não efeito
```

**Framework Senior:**
1. **Identificar confounders** (variáveis de confusão)
2. **Testar hipóteses reversas** ("e se a relação for inversa?")
3. **Usar linguagem cautelosa** ("associado com" vs. "causa")
4. **Propor testes A/B** para validar causalidade

---

### **6. Escala e Performance em Produção**

**Cuidados com código "notebook-friendly" que quebra em produção:**

```python
# ❌ Código de notebook (funciona com 500k linhas)
df['new_feature'] = df.apply(lambda x: complex_function(x), axis=1)
# Tempo: 45 minutos

# ✅ Código production-ready
df['new_feature'] = df.groupby('customer_id')['value'].transform('sum')
# Tempo: 3 segundos
```

**Checklist de produção:**
- [ ] Código vetorizado (evitar loops quando possível)
- [ ] Memória gerenciada (usar chunks para dados grandes)
- [ ] Features replicáveis (sem random seeds não controlados)
- [ ] Pipeline serializado (salvar preprocessors com modelo)

---

## 📋 **CHECKLIST DE DECISÕES CRÍTICAS**

### **Antes de começar:**
- [ ] Defini período de análise e justificativa?
- [ ] Mapeei todas as fontes de dados faltantes?
- [ ] Alinhe definição de "cliente ativo" com stakeholders?
- [ ] Estabeleci baseline para comparar modelos?

### **Durante análise:**
- [ ] Documentei TODAS decisões de limpeza de dados?
- [ ] Validei premissas estatísticas (normalidade, etc.)?
- [ ] Testei robustez das conclusões (sensitivity analysis)?
- [ ] Criei visualizações que executivos entendam?

### **Antes de entregar:**
- [ ] Resultados fazem sentido de NEGÓCIO (não só estatístico)?
- [ ] Consegui explicar em linguagem não-técnica?
- [ ] Identifiquei limitações e próximos passos?
- [ ] Código está reproduzível (requirements.txt, seeds, etc.)?

---

## 🎓 **PRINCÍPIOS DE UM SENIOR DATA SCIENTIST**

### **1. Ceticismo Saudável**
```python
# Sempre pergunte:
"Este resultado é BOM DEMAIS para ser verdade?"
"Estou vendo padrão real ou ruído?"
"E se eu estiver errado? Qual o impacto?"
```

### **2. Transparência Radical**
```python
# Documente incertezas
model_report = {
    "confidence": "Média-Alta",
    "limitations": [
        "Dataset não inclui clientes B2C",
        "Período de análise limitado (12 meses)",
        "Sem dados de marketing campaigns"
    ],
    "assumptions": [
        "Comportamento futuro similar ao passado",
        "CustomerID mapping está correto"
    ]
}
```

### **3. Business Outcome First**
```python
# Sempre alinhe com métrica de negócio
if technical_metric_improved and not business_metric_improved:
    print("⚠️ MODELO INÚTIL - revisar abordagem")
```

### **4. Código como Comunicação**
```python
# ❌ Código de junior
df2 = df1[df1['col3'] > df1['col2'].quantile(0.9)]

# ✅ Código de senior
high_value_customers = customers[
    customers['lifetime_value'] > customers['lifetime_value'].quantile(0.9)
]
# Nomes descritivos > comentários
```

---

## 🚨 **RED FLAGS para Abortar/Pivotar**

**Pare e reavalie se:**
1. **>40% dos dados precisam ser descartados** → dataset pode ser inadequado
2. **Segmentos têm overlap >60%** → clustering não está encontrando padrões reais
3. **Modelo performa igual ao baseline** → features não têm poder preditivo
4. **Stakeholders não entendem resultados** → comunicação falhou
5. **Recomendações não são acionáveis** → análise é academicamente interessante, mas inútil

---

## ✅ **RESUMO EXECUTIVO: O que fazer AGORA**

### **Antes de codar 1 linha:**

1. **Baixar e fazer scan inicial do dataset**
   - Row count, column types, missing %
   - Identificar surpresas cedo

2. **Definir 3-5 perguntas de negócio específicas**
   - "Quais segmentos têm maior potencial de crescimento?"
   - "Qual timing ideal para reengajar clientes inativos?"
   - "Quais produtos devemos bundling juntos?"

3. **Criar documento de decisões**
   - Markdown file: `DECISIONS.md`
   - Registrar cada escolha metodológica e POR QUÊ

4. **Setup de ambiente reproduzível**
   ```bash
   # requirements.txt com versões fixas
   pandas==2.0.3
   scikit-learn==1.3.0
   # etc
   ```

---

#!/bin/bash
#
# Script para coletar métricas de CPU de um pod Spring Boot via Actuator
# e calcular requests.cpu e targetAverageUtilization para o HPA.
#
# Uso:
#   ./calc-hpa-cpu.sh <ACTUATOR_URL> <ITERACOES> <INTERVALO_SEG>
#
# Exemplo:
#   ./calc-hpa-cpu.sh http://localhost:8080 20 5
#
#   Isso fará 20 coletas, uma a cada 5 segundos.

ACTUATOR_URL="$1"
ITER="$2"
SLEEP_SEC="$3"

if [[ -z "$ACTUATOR_URL" || -z "$ITER" || -z "$SLEEP_SEC" ]]; then
  echo "Uso: $0 <ACTUATOR_URL> <ITERACOES> <INTERVALO_SEG>"
  exit 1
fi

# Lista para armazenar valores coletados
values=()

echo "📡 Coletando métricas de CPU de $ACTUATOR_URL por $ITER iterações a cada $SLEEP_SEC segundos..."
echo

for ((i=1; i<=ITER; i++)); do
  cpu_usage=$(curl -s "$ACTUATOR_URL/actuator/metrics/process.cpu.usage" \
    | jq -r '.measurements[0].value')

  if [[ "$cpu_usage" == "null" || -z "$cpu_usage" ]]; then
    echo "❌ Não foi possível obter métricas. Verifique o Actuator."
    exit 1
  fi

  # Converte para milicpu
  mcpu=$(echo "$cpu_usage * 1000" | bc -l)
  mcpu_int=$(printf "%.0f" "$mcpu")

  values+=("$mcpu_int")
  echo "[$i] Uso de CPU: $mcpu_int mCPU"

  sleep "$SLEEP_SEC"
done

# Calcula média e pico
cpu_avg=$(echo "${values[@]}" | tr ' ' '\n' | awk '{sum+=$1} END {printf "%.0f", sum/NR}')
cpu_peak=$(printf '%s\n' "${values[@]}" | sort -nr | head -n1)

# Fórmulas
SAFETY_FACTOR=1.25
REQUESTS_CPU=$(echo "$cpu_peak * $SAFETY_FACTOR" | bc -l | awk '{printf "%.0f", $1}')
TARGET_PERCENT=$(echo "($cpu_avg / $REQUESTS_CPU) * 100 * 1.1" | bc -l | awk '{printf "%.0f", $1}')

echo
echo "📊 Resultado:"
echo "-------------------------------"
echo "Média CPU:        ${cpu_avg}m"
echo "Pico CPU:         ${cpu_peak}m"
echo "Requests sugerido: ${REQUESTS_CPU}m"
echo "Target HPA suger.: ${TARGET_PERCENT}%"
echo "-------------------------------"


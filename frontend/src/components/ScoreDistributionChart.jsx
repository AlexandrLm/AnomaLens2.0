import React, { useMemo } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import annotationPlugin from 'chartjs-plugin-annotation';
import { Bar } from 'react-chartjs-2';
import {
  Box,
  Typography,
  CircularProgress,
  Alert,
  useTheme
} from '@mui/material';
import { alpha } from '@mui/material/styles';

// Регистрация необходимых компонентов Chart.js
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  annotationPlugin
);

// Функция для создания данных гистограммы
const createHistogramData = (scores, numBins = 20) => {
  if (!scores || scores.length === 0) {
    return { labels: [], datasets: [] };
  }

  const minScore = Math.min(...scores);
  const maxScore = Math.max(...scores);

  // Если все значения одинаковые, создаем один бин
  if (minScore === maxScore) {
      const label = `${minScore.toFixed(3)}`;
      return {
          labels: [label],
          datasets: [
              {
                  label: 'Частота',
                  data: [scores.length],
                  backgroundColor: 'rgba(54, 162, 235, 0.6)',
                  borderColor: 'rgba(54, 162, 235, 1)',
                  borderWidth: 1,
              },
          ],
      };
  }

  const binWidth = (maxScore - minScore) / numBins;
  const bins = Array(numBins).fill(0);
  const labels = [];

  // Создаем метки для бинов
  for (let i = 0; i < numBins; i++) {
    const binStart = minScore + i * binWidth;
    const binEnd = binStart + binWidth;
    // Форматируем метки для лучшей читаемости
    labels.push(`${binStart.toFixed(3)} - ${binEnd.toFixed(3)}`);
  }

  // Распределяем скоры по бинам
  scores.forEach(score => {
    // Определяем индекс бина. Добавляем малое значение к maxScore для включения его в последний бин
    let binIndex = Math.floor((score - minScore) / binWidth);
    // Корректируем индекс для максимального значения
    if (score === maxScore) {
        binIndex = numBins - 1;
    } 
    // Убедимся что индекс в границах (на случай ошибок округления)
    binIndex = Math.max(0, Math.min(numBins - 1, binIndex)); 
    bins[binIndex]++;
  });

  return {
    labels,
    datasets: [
      {
        label: 'Частота',
        data: bins,
        backgroundColor: 'rgba(75, 192, 192, 0.6)', // Цвет для гистограммы
        borderColor: 'rgba(75, 192, 192, 1)',
        borderWidth: 1,
      },
    ],
  };
};

function ScoreDistributionChart({ scores, detectorName, isLoading, error, thresholdValue }) {
  const theme = useTheme();

  // Используем useMemo для мемоизации данных гистограммы
  const histogramData = useMemo(() => {
      if (isLoading || error || !scores) return { labels: [], datasets: [] };
      // Можно настроить количество бинов
      return createHistogramData(scores, 25); 
  }, [scores, isLoading, error]);

  const options = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false, // Легенда не очень нужна для одной серии данных
      },
      title: {
        display: true,
        text: `Распределение Оценок Аномальности (${detectorName || 'N/A'})`,
        font: {
            size: 16,
            weight: 'bold',
        },
        padding: {
            top: 10,
            bottom: 20
        }
      },
      tooltip: {
        callbacks: {
            // Можно настроить тултипы, если нужно
        }
      },
      annotation: {
        annotations: {
          ...(thresholdValue !== null && thresholdValue !== undefined && {
            thresholdLine: {
              type: 'line',
              scaleID: 'x',
              value: thresholdValue,
              borderColor: theme.palette.error.main, // Красный цвет для порога
              borderWidth: 2,
              borderDash: [6, 6], // Пунктирная линия
              label: {
                content: `Порог (${thresholdValue.toFixed(3)})`,
                enabled: true,
                position: 'start', // Положение метки
                backgroundColor: alpha(theme.palette.error.main, 0.8),
                color: theme.palette.error.contrastText,
                font: {
                    size: 10
                },
                yAdjust: -10, // Сдвиг метки вверх
              }
            }
          })
        }
      }
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Диапазон Оценок (Anomaly Score)',
        },
         ticks: {
             // Попробуем показывать меньше меток на оси X, если их много
             autoSkip: true,
             maxTicksLimit: 10 
         }
      },
      y: {
        title: {
          display: true,
          text: 'Частота',
        },
        beginAtZero: true, // Начинаем ось Y с нуля
      },
    },
  }), [detectorName, thresholdValue, theme]);

  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 300 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 300 }}>
        <Alert severity="error" sx={{ width: '100%' }}>{error}</Alert>
      </Box>
    );
  }

  if (!scores || scores.length === 0) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 300 }}>
        <Typography color="text.secondary">Нет данных о скорах для отображения.</Typography>
      </Box>
    );
  }

  return (
    // Задаем высоту контейнера, чтобы график имел размер
    <Box sx={{ height: 350, width: '100%', p: 1 }}>
      <Bar options={options} data={histogramData} />
    </Box>
  );
}

export default ScoreDistributionChart; 
import React, { useMemo } from 'react';
import {
    Chart as ChartJS,
    LinearScale,
    PointElement,
    LineElement, // Хотя мы используем scatter, LineElement нужен для тултипов/легенды
    Tooltip,
    Legend,
} from 'chart.js';
import zoomPlugin from 'chartjs-plugin-zoom';
import 'hammerjs'; // Для поддержки touch-жестов
import { Scatter } from 'react-chartjs-2';
import {
    Box, Typography, CircularProgress, Alert, useTheme
} from '@mui/material';
import { alpha } from '@mui/material/styles';

ChartJS.register(
    LinearScale, // Scatter использует LinearScale для обеих осей
    PointElement,
    LineElement,
    Tooltip,
    Legend
);

// Отдельно регистрируем плагин зума
ChartJS.register(zoomPlugin);

function FeatureScatterPlot({ data, featureXName, featureYName }) {
    const theme = useTheme();

    const chartData = useMemo(() => {
        if (!data || data.length === 0) {
            return { datasets: [] };
        }

        // Разделяем данные на нормальные и аномальные
        const normalPoints = data.filter(p => !p.is_anomaly).map(p => ({ x: p.x, y: p.y, id: p.id }));
        const anomalyPoints = data.filter(p => p.is_anomaly).map(p => ({ x: p.x, y: p.y, id: p.id }));

        return {
            datasets: [
                {
                    label: 'Нормальные',
                    data: normalPoints,
                    backgroundColor: alpha(theme.palette.primary.main, 0.6),
                    borderColor: theme.palette.primary.dark,
                    pointRadius: 5,
                    pointHoverRadius: 7,
                },
                {
                    label: 'Аномалии',
                    data: anomalyPoints,
                    backgroundColor: alpha(theme.palette.error.main, 0.7),
                    borderColor: theme.palette.error.dark,
                    pointRadius: 6, // Чуть больше для выделения
                    pointHoverRadius: 8,
                },
            ],
        };
    }, [data, theme]);

    const options = useMemo(() => ({
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'top',
            },
            title: {
                display: true,
                text: `Диаграмма Рассеяния: ${featureYName} vs ${featureXName}`,
                font: {
                    size: 16
                },
                 padding: {
                    bottom: 15
                }
            },
            tooltip: {
                callbacks: {
                    label: function(context) {
                        const point = context.raw;
                        const datasetLabel = context.dataset.label || 'Точка';
                        if (!point) return datasetLabel;
                        
                        const xLabel = context.chart.options.scales.x.title.text || 'X';
                        const yLabel = context.chart.options.scales.y.title.text || 'Y';
                        
                        const xValueFormatted = typeof point.x === 'number' ? point.x.toLocaleString() : point.x; 
                        const yValueFormatted = typeof point.y === 'number' ? point.y.toLocaleString() : point.y;
                        
                        return `${datasetLabel} (ID: ${point.id}): ${xLabel}=${xValueFormatted}, ${yLabel}=${yValueFormatted}`;
                    }
                }
            },
            zoom: {
                pan: {
                    enabled: true,
                    mode: 'xy', // Разрешаем панорамирование по обеим осям
                    threshold: 5, // Порог срабатывания (пиксели)
                },
                zoom: {
                    wheel: { // Масштабирование колесиком мыши
                        enabled: true,
                    },
                    pinch: { // Масштабирование щипком (touch)
                        enabled: true
                    },
                    mode: 'xy', // Разрешаем масштабирование по обеим осям
                },
                 limits: {
                    // Можно задать ограничения на масштабирование, если нужно
                    // x: {min: 0, max: 1000000},
                    // y: {min: 0, max: 10}
                 }
            }
        },
        scales: {
            x: {
                title: {
                    display: true,
                    text: featureXName.replace(/_/g, ' ') // Заменяем _ на пробелы для читаемости
                },
                type: 'linear', // Явно указываем тип оси
                position: 'bottom',
                ticks: {
                    callback: function(value, index, ticks) {
                        if (Math.abs(value) >= 1e6) return (value / 1e6).toFixed(1) + 'M';
                        if (Math.abs(value) >= 1e3) return (value / 1e3).toFixed(0) + 'K';
                        return value; // Возвращаем как есть для меньших чисел
                    }
                }
            },
            y: {
                title: {
                    display: true,
                    text: featureYName.replace(/_/g, ' ')
                },
                type: 'linear',
                ticks: {
                    callback: function(value, index, ticks) {
                        if (Math.abs(value) >= 1e6) return (value / 1e6).toFixed(1) + 'M';
                        if (Math.abs(value) >= 1e3) return (value / 1e3).toFixed(0) + 'K';
                        return value;
                    }
                }
            }
        }
    }), [featureXName, featureYName]);

    return (
        // Используем 100% высоты родителя (Paper)
        <Box sx={{ height: '100%', width: '100%' }}> 
            <Scatter options={options} data={chartData} />
        </Box>
    );
}

export default FeatureScatterPlot; 
import React, { useMemo } from 'react';
import {
    Chart as ChartJS,
    LinearScale,
    PointElement,
    LineElement, // Хотя мы используем scatter, LineElement нужен для тултипов/легенды
    Tooltip,
    Legend,
} from 'chart.js';
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
                        let label = context.dataset.label || '';
                        if (label) {
                            label += ': ';
                        }
                        const point = context.raw;
                        if (point) {
                            label += `ID ${point.id}, X: ${point.x}, Y: ${point.y}`;
                        }
                        return label;
                    }
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
                position: 'bottom'
            },
            y: {
                title: {
                    display: true,
                    text: featureYName.replace(/_/g, ' ')
                },
                type: 'linear'
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
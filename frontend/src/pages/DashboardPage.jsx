import React, { useState, useEffect, useCallback, useMemo } from 'react';
import axios from 'axios';
import {
    Typography, Paper, Grid, CircularProgress, Alert, Card, CardContent, Button,
    Box, List, ListItem, ListItemText, Divider, Chip, Stack, useTheme, IconButton, ListItemIcon, Collapse
} from '@mui/material';
import { Train, Search, ExpandLess, ExpandMore, ErrorOutline, CheckCircleOutline, Settings, WarningAmberOutlined, InfoOutlined } from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
// ИМПОРТЫ ДЛЯ ГРАФИКОВ
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    BarElement, // Для Bar chart
    Title,
    Tooltip,
    Legend,
} from 'chart.js';
import { Line, Bar } from 'react-chartjs-2'; // Импортируем типы графиков
import { alpha } from '@mui/material/styles';

// Регистрируем компоненты Chart.js
ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    BarElement,
    Title,
    Tooltip,
    Legend
);
// --------------------------

// Helper to format date
const formatDate = (dateString) => {
    try {
        return new Date(dateString).toLocaleString();
    } catch (e) {
        return dateString;
    }
};

// ВОССТАНАВЛИВАЕМ getAnomalySettings
const getAnomalySettings = () => {
    const savedSettings = localStorage.getItem('anomalySettings');
    const defaults = {
        limit: 10000,
        z_threshold: 3.0,
        dbscan_eps: 0.5,
        dbscan_min_samples: 5,
    };
    if (savedSettings) {
        try {
            const parsed = JSON.parse(savedSettings);
            return {
                limit: parseInt(parsed.limit, 10) || defaults.limit,
                z_threshold: parseFloat(parsed.z_threshold) || defaults.z_threshold,
                dbscan_eps: parseFloat(parsed.dbscan_eps) || defaults.dbscan_eps,
                dbscan_min_samples: parseInt(parsed.dbscan_min_samples, 10) || defaults.dbscan_min_samples,
            };
        } catch (error) {
            console.error("Failed to parse settings from localStorage:", error);
            return defaults;
        }
    }
    return defaults;
};

function DashboardPage() {
    const theme = useTheme();
    const navigate = useNavigate();

    // Состояние для сводки
    const [summary, setSummary] = useState(null);
    const [loadingSummary, setLoadingSummary] = useState(true);
    const [errorSummary, setErrorSummary] = useState('');

    // Состояние для аномалий
    const [anomalies, setAnomalies] = useState([]);
    const [loadingAnomalies, setLoadingAnomalies] = useState(true);
    const [errorAnomalies, setErrorAnomalies] = useState('');

    // Состояние для процесса детекции
    const [detecting, setDetecting] = useState(false);
    const [detectionResult, setDetectionResult] = useState(null);
    const [detectionError, setDetectionError] = useState('');

    // --- НОВОЕ: Состояние для процесса обучения/подготовки ---
    const [training, setTraining] = useState(false);
    const [trainingResult, setTrainingResult] = useState(null);
    const [trainingError, setTrainingError] = useState('');
    // ---------------------------------------------------------

    // State variables
    const [loadingTrain, setLoadingTrain] = useState(false);
    const [trainError, setTrainError] = useState('');
    const [trainSuccess, setTrainSuccess] = useState('');

    const [loadingDetectActivity, setLoadingDetectActivity] = useState(false);
    const [loadingDetectOrder, setLoadingDetectOrder] = useState(false);
    const [detectError, setDetectError] = useState('');
    const [openAnomalyId, setOpenAnomalyId] = useState(null);

    // State for the new Order detection button
    const [loadingOrderDetect, setLoadingOrderDetect] = useState(false);
    const [orderDetectError, setOrderDetectError] = useState('');

    // НОВЫЕ СОСТОЯНИЯ ДЛЯ ГРАФИКОВ
    const [activityTimelineData, setActivityTimelineData] = useState(null);
    const [loadingTimeline, setLoadingTimeline] = useState(true);
    const [errorTimeline, setErrorTimeline] = useState('');
    const [anomalySummaryData, setAnomalySummaryData] = useState(null);
    const [loadingAnomalySummary, setLoadingAnomalySummary] = useState(true);
    const [errorAnomalySummary, setErrorAnomalySummary] = useState('');
    // ---------------------------------------------------------

    // --- Подготовка данных и опций для графиков (используем useMemo) ---
    const activityTimelineChartData = useMemo(() => {
        if (!activityTimelineData) return null;
        const labels = activityTimelineData.map(item => {
            // Форматируем метку в зависимости от интервала (может потребоваться улучшение)
            try {
                 return new Date(item.interval).toLocaleDateString(); // Пока просто дата
            } catch { return item.interval; } // Фоллбэк
        });
        const dataPoints = activityTimelineData.map(item => item.count);

        return {
            labels,
            datasets: [
                {
                    label: 'Активность пользователей',
                    data: dataPoints,
                    borderColor: theme.palette.primary.main,
                    backgroundColor: theme.palette.primary.light + '80', // с прозрачностью
                    tension: 0.1,
                    fill: true,
                },
            ],
        };
    }, [activityTimelineData, theme.palette.primary]);

    const anomalySummaryChartData = useMemo(() => {
        if (!anomalySummaryData) return null;
        const labels = anomalySummaryData.map(item => item.detector_name);
        const dataPoints = anomalySummaryData.map(item => item.count);

        return {
            labels,
            datasets: [
                {
                    label: 'Кол-во аномалий',
                    data: dataPoints,
                    backgroundColor: [
                        theme.palette.error.light + 'b3', // Пример разных цветов
                        theme.palette.warning.light + 'b3',
                        theme.palette.info.light + 'b3',
                        theme.palette.secondary.light + 'b3',
                    ],
                    borderColor: [
                       theme.palette.error.main,
                       theme.palette.warning.main,
                       theme.palette.info.main,
                       theme.palette.secondary.main,
                    ],
                    borderWidth: 1,
                },
            ],
        };
    }, [anomalySummaryData, theme.palette]);

    const chartOptions = useMemo(() => ({
        responsive: true,
        maintainAspectRatio: false, // Позволяет задавать высоту через контейнер
        plugins: {
            legend: {
                position: 'top',
            },
            title: {
                display: true,
                // Текст заголовка будет задан отдельно для каждого графика
            },
        },
        scales: {
             y: {
                 beginAtZero: true
             }
        }
    }), []);
    // ------------------------------------------------------------------

    // Функция для загрузки сводки
    const fetchDashboardData = useCallback(async () => {
        setLoadingSummary(true);
        setErrorSummary('');
        try {
            const response = await axios.get('/api/dashboard/summary');
            setSummary(response.data);
        } catch (err) {
            console.error("Failed to fetch dashboard summary:", err);
            let errorMessage = 'Ошибка при загрузке сводки.';
            if (err.response && err.response.status === 404) {
                errorMessage = 'API сводки (/api/dashboard/summary) не найдено.';
            } else if (err.message) {
                errorMessage += ` ${err.message}`;
            }
            setErrorSummary(errorMessage);
            setSummary(null);
        } finally {
            setLoadingSummary(false);
        }
    }, []);

    // Функция для загрузки аномалий
    const fetchAnomalies = useCallback(async (showLoader = true) => {
        if (showLoader) setLoadingAnomalies(true);
        setDetectError(''); // Clear previous detection errors on refresh
        try {
            const response = await axios.get('/api/anomalies/?limit=100'); // Fetch last 100
            setAnomalies(response.data || []);
        } catch (error) {
            console.error("Error fetching anomalies:", error);
            setDetectError('Не удалось загрузить список аномалий.');
            setAnomalies([]);
        } finally {
            if (showLoader) setLoadingAnomalies(false);
        }
    }, []);

    // НОВЫЕ ФУНКЦИИ ЗАГРУЗКИ ДАННЫХ ДЛЯ ГРАФИКОВ
    const fetchActivityTimeline = useCallback(async (interval = 'day') => {
        setLoadingTimeline(true);
        setErrorTimeline('');
        try {
            const response = await axios.get(`/api/charts/activity_timeline?interval=${interval}`);
            setActivityTimelineData(response.data);
        } catch (error) {
            console.error("Error fetching activity timeline:", error);
            setErrorTimeline('Не удалось загрузить данные для графика активности.');
            setActivityTimelineData(null);
        }
        finally {
             setLoadingTimeline(false);
        }
    }, []);

    const fetchAnomalySummary = useCallback(async () => {
        setLoadingAnomalySummary(true);
        setErrorAnomalySummary('');
        try {
            const response = await axios.get('/api/charts/anomaly_summary');
            setAnomalySummaryData(response.data);
        } catch (error) {
            console.error("Error fetching anomaly summary:", error);
            setErrorAnomalySummary('Не удалось загрузить данные для сводки по аномалиям.');
            setAnomalySummaryData(null);
        }
        finally {
            setLoadingAnomalySummary(false);
        }
    }, []);
    // ---------------------------------------------------------

    // Загрузка всех данных при монтировании
    useEffect(() => {
        fetchDashboardData();
        fetchAnomalies();
        fetchActivityTimeline(); // Загружаем график активности (по дням по умолчанию)
        fetchAnomalySummary();   // Загружаем сводку аномалий
    }, [fetchDashboardData, fetchAnomalies, fetchActivityTimeline, fetchAnomalySummary]);

    // ВОЗВРАЩАЕМ логику Train Models с payload
    const handleTrainModels = async () => {
        setLoadingTrain(true);
        setTrainError('');
        setTrainSuccess('');
        setDetectionResult(null);
        try {
            const currentSettings = getAnomalySettings(); // Используем localStorage
            const payload = {
                limit: currentSettings.limit,
                z_threshold: currentSettings.z_threshold,
                // entity_type опционален, бэкенд обработает
            };
            console.log("Sending training payload:", JSON.stringify(payload, null, 2));
            const response = await axios.post('/api/anomalies/train', payload); // Снова передаем payload
            setTrainSuccess(response.data.message || 'Обучение/подготовка успешно запущены.');
        } catch (error) {
            console.error("Error training models:", error);
            const errorMsg = error.response?.data?.detail || error.message || 'Произошла ошибка.';
            setTrainError(`Ошибка при запуске обучения: ${errorMsg}`);
        } finally {
            setLoadingTrain(false);
        }
    };

    // ВОЗВРАЩАЕМ логику Detect ACTIVITY Anomalies с payload
    const handleDetectActivityAnomalies = async () => {
        setLoadingDetectActivity(true);
        setDetectionResult(null);
        setDetectionError('');
        setTrainingResult(null); // Сбрасываем результат обучения
        setTrainingError('');

        const currentSettings = getAnomalySettings();
        console.log("Running activity detection with settings:", currentSettings);

        try {
            const response = await axios.post('/api/anomalies/detect', {
                entity_type: 'activity',
                limit: currentSettings.limit,
                z_threshold: currentSettings.z_threshold,
                dbscan_eps: currentSettings.dbscan_eps,
                dbscan_min_samples: currentSettings.dbscan_min_samples,
                algorithms: ['statistical_zscore', 'isolation_forest', 'dbscan'],
            });
            setDetectionResult(response.data);
            fetchAnomalies(false); // Refresh anomaly list without showing loader
        } catch (error) {
            console.error("Error detecting activity anomalies:", error);
            setDetectionError(error.response?.data?.detail || 'Ошибка при запуске поиска аномалий активностей.');
        } finally {
            setLoadingDetectActivity(false);
        }
    };

    // ВОЗВРАЩАЕМ логику Detect ORDER Anomalies с payload
    const handleDetectOrderAnomalies = async () => {
        setLoadingDetectOrder(true);
        setDetectionResult(null);
        setDetectionError('');
        setTrainingResult(null);
        setTrainingError('');

        const currentSettings = getAnomalySettings();
        console.log("Running order detection with settings:", currentSettings);

        try {
            const response = await axios.post('/api/anomalies/detect', {
                entity_type: 'order',
                algorithms: ['statistical_zscore'], // Только стат. для заказов
                limit: currentSettings.limit,
                z_threshold: currentSettings.z_threshold,
                // Параметры DBSCAN не передаем, т.к. он не используется для заказов
            });
            setDetectionResult(response.data);
            fetchAnomalies(false);
        } catch (error) {
            console.error("Error detecting order anomalies:", error);
            setDetectionError(error.response?.data?.detail || 'Ошибка при запуске поиска аномалий заказов.');
        } finally {
            setLoadingDetectOrder(false);
        }
    };

    // Toggle anomaly details
    const handleToggleAnomaly = (id) => {
        setOpenAnomalyId(openAnomalyId === id ? null : id);
    };

    // Navigate to Settings page
    const goToSettings = () => {
        navigate('/settings');
    };

    // Combined loading state for disabling buttons
    const isDetecting = loadingDetectActivity || loadingDetectOrder;

    return (
        <Grid container spacing={3}>
            {/* Сводка */}
            <Grid item xs={12}>
                <Paper elevation={3} sx={{ p: 3 }}>
                    <Typography variant="h4" gutterBottom>
                        Панель Управления
                    </Typography>

                    {loadingSummary && <CircularProgress size={24} sx={{ mr: 1 }} />}
                    {errorSummary && <Alert severity="error" sx={{ mt: 1 }}>{errorSummary}</Alert>}

                    {summary && (
                        <Grid container spacing={2} sx={{ mt: 1 }}>
                            {Object.entries(summary).map(([key, value]) => (
                                <Grid item xs={12} sm={6} md={4} lg={2.4} key={key}> {/* Adjust lg for 5 items */}
                                    <Card>
                                        <CardContent>
                                            <Typography variant="h6" component="div" sx={{ textTransform: 'capitalize', fontSize: '0.9rem' }}>
                                                {key.replace(/_/g, ' ')}
                                            </Typography>
                                            <Typography variant="h5" color="text.secondary">
                                                {typeof value === 'number' ? value.toLocaleString() : value}
                                            </Typography>
                                        </CardContent>
                                    </Card>
                                </Grid>
                            ))}
                        </Grid>
                    )}
                </Paper>
            </Grid>

            {/* НОВАЯ СЕКЦИЯ ДЛЯ ГРАФИКОВ */}
            <Grid item xs={12} md={7}> {/* Занимает большую часть ширины */}
                <Paper elevation={3} sx={{ p: 2, height: '400px' }}>
                    <Typography variant="h6" gutterBottom>Активность пользователей (по дням)</Typography>
                    {loadingTimeline && <CircularProgress sx={{ display: 'block', margin: 'auto' }} />}
                    {errorTimeline && <Alert severity="error">{errorTimeline}</Alert>}
                    {activityTimelineChartData && !loadingTimeline && (
                        <Box sx={{ height: 'calc(100% - 40px)' }}> {/* Вычитаем высоту заголовка */}
                            <Line
                                options={{
                                    ...chartOptions,
                                    plugins: {
                                         ...chartOptions.plugins,
                                         title: { ...chartOptions.plugins.title, text: 'Динамика активности' }
                                    }
                                }}
                                data={activityTimelineChartData}
                            />
                        </Box>
                    )}
                </Paper>
            </Grid>
            <Grid item xs={12} md={5}> {/* Занимает оставшуюся часть */}
                 <Paper elevation={3} sx={{ p: 2, height: '400px' }}>
                    <Typography variant="h6" gutterBottom>Сводка по Аномалиям</Typography>
                    {loadingAnomalySummary && <CircularProgress sx={{ display: 'block', margin: 'auto' }} />}
                    {errorAnomalySummary && <Alert severity="error">{errorAnomalySummary}</Alert>}
                    {anomalySummaryChartData && !loadingAnomalySummary && (
                         <Box sx={{ height: 'calc(100% - 40px)' }}>
                            <Bar
                                options={{
                                     ...chartOptions,
                                     plugins: {
                                         ...chartOptions.plugins,
                                         title: { ...chartOptions.plugins.title, text: 'Аномалии по детекторам' },
                                         legend: { display: false } // Легенда не нужна для одного датасета
                                    },
                                     indexAxis: 'y',
                                }}
                                data={anomalySummaryChartData}
                            />
                        </Box>
                    )}
                </Paper>
            </Grid>
            {/* --------------------------------- */}

            {/* Управление и Отображение Аномалий */}
            <Grid item xs={12}>
                <Paper elevation={3} sx={{ p: 3 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2, flexWrap: 'wrap', gap: 2 }}>
                        <Typography variant="h5" gutterBottom sx={{ mb: 0 }}>
                            Обнаруженные Аномалии
                        </Typography>
                        <Stack direction="row" spacing={2} flexWrap="wrap" useFlexGap> {/* Добавляем wrap и useFlexGap */}
                            <Button
                                variant="outlined" // Сделаем кнопку обучения отличающейся
                                onClick={handleTrainModels}
                                disabled={training || isDetecting} // Блокируем во время любого процесса
                            >
                                {training ? <CircularProgress size={24} sx={{ mr: 1 }} /> : null}
                                Обучить / Подготовить Модели
                            </Button>
                            <Button
                                variant="contained"
                                onClick={handleDetectActivityAnomalies}
                                disabled={isDetecting || training} // Блокируем во время любого процесса
                            >
                                {loadingDetectActivity ? <CircularProgress size={24} sx={{ mr: 1 }} /> : null}
                                Запустить Поиск Аномалий
                            </Button>
                            {/* --- НОВАЯ КНОПКА --- */}
                            <Button variant="contained" color="secondary" onClick={handleDetectOrderAnomalies} disabled={isDetecting || training}>
                                {loadingDetectOrder ? <CircularProgress size={24} sx={{ mr: 1 }} /> : null}
                                Поиск (Заказы)
                            </Button>
                            {/* ------------------ */}
                        </Stack>
                    </Box>

                    {/* --- НОВОЕ: Результат/ошибка обучения --- */}
                    {trainingError && <Alert severity="error" sx={{ mb: 2 }}>{trainingError}</Alert>}
                    {trainingResult && (
                        <Alert severity="info" sx={{ mb: 2 }}>
                            {trainingResult.message}
                        </Alert>
                    )}
                    {/* ------------------------------------ */}

                    {/* Результат/ошибка детекции */}
                    {detectionError && <Alert severity="error" sx={{ mb: 2 }}>{detectionError}</Alert>}
                    {detectionResult && (
                        <Alert severity="success" sx={{ mb: 2 }}>
                            {detectionResult.message}
                            <ul>
                                {Object.entries(detectionResult.anomalies_saved_by_algorithm || {}).map(([algo, count]) => (
                                    <li key={algo}>{`${algo}: сохранено ${count} аномалий`}</li>
                                ))}
                            </ul>
                        </Alert>
                    )}

                    {/* Список аномалий */}
                    {loadingAnomalies && <CircularProgress sx={{ display: 'block', margin: '20px auto' }} />}
                    {errorAnomalies && <Alert severity="warning" sx={{ mt: 2 }}>{errorAnomalies}</Alert>}

                    {!loadingAnomalies && !errorAnomalies && anomalies.length === 0 && (
                        <Alert severity="info" sx={{ mt: 2 }}>Аномалии не найдены или еще не обнаружены.</Alert>
                    )}

                    {!loadingAnomalies && anomalies.length > 0 && (
                        <List sx={{ width: '100%', bgcolor: 'background.paper', mt: 2, borderRadius: 2, overflow: 'hidden' }}>
                            {anomalies.map((anomaly) => {
                                // Форматируем timestamp
                                const detectionTime = anomaly.detected_at // Используем новое поле
                                    ? new Date(anomaly.detected_at).toLocaleString()
                                    : 'N/A';
                                
                                // Определяем цвет и иконку для severity
                                let severityColor = 'default';
                                let severityIcon = <CheckCircleOutline fontSize="small" />;
                                const severityText = (anomaly.severity || 'Medium').toUpperCase();
                                
                                if (severityText === 'HIGH') {
                                    severityColor = 'error';
                                    severityIcon = <ErrorOutline fontSize="small" color="error" />;
                                } else if (severityText === 'MEDIUM') {
                                    severityColor = 'warning';
                                    severityIcon = <WarningAmberOutlined fontSize="small" color="warning" />;
                                } else if (severityText === 'LOW') {
                                    severityColor = 'info';
                                    severityIcon = <InfoOutlined fontSize="small" color="info" />;
                                }

                                let primaryText = (
                                    <Stack direction="row" spacing={1} alignItems="center">
                                         <Chip 
                                            label={severityText}
                                            color={severityColor}
                                            size="small"
                                            icon={severityIcon}
                                            sx={{ fontWeight: 500, height: 22, '.MuiChip-label': { fontSize: '0.75rem' } }}
                                        />
                                        <Typography variant="body2" component="span">
                                            {`${anomaly.detector_name} | ${anomaly.entity_type.toUpperCase()} ID: ${anomaly.entity_id} | ${detectionTime}`}
                                        </Typography>
                                    </Stack>
                                );

                                // Парсим детали для улучшения отображения
                                let detailsContent = null;
                                try {
                                    if (anomaly.details) {
                                        const detailsObj = typeof anomaly.details === 'string' 
                                                            ? JSON.parse(anomaly.details) 
                                                            : anomaly.details; // Already an object

                                        // Форматируем детали
                                        detailsContent = (
                                            <List dense disablePadding sx={{ pl: 2 }}>
                                                {Object.entries(detailsObj).map(([key, value]) => {
                                                    let displayValue = value;
                                                    if (typeof value === 'object' && value !== null) {
                                                         displayValue = <pre style={{ margin: 0, fontSize: '0.75rem' }}>{JSON.stringify(value, null, 2)}</pre>;
                                                    } else if (typeof value === 'number') {
                                                        displayValue = value.toFixed(4);
                                                    }

                                                    return (
                                                        <ListItem key={key} sx={{ pl: 0, pr: 0, pt: 0.5, pb: 0.5 }}>
                                                            <ListItemText 
                                                                primaryTypographyProps={{ variant: 'caption', fontWeight: 'bold' }} 
                                                                primary={`${key}:`}
                                                                secondaryTypographyProps={{ variant: 'caption', component: 'div', whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}
                                                                secondary={displayValue}
                                                                sx={{ m: 0 }}
                                                            />
                                                        </ListItem>
                                                    );
                                                })}
                                            </List>
                                        );
                                    } else {
                                         detailsContent = <Typography variant="caption">Детали отсутствуют.</Typography>;
                                    }
                                } catch (e) {
                                    console.error("Error parsing anomaly details:", e, anomaly.details);
                                    detailsContent = <Typography variant="caption" color="error">Ошибка отображения деталей.</Typography>;
                                    if (anomaly.details) {
                                         detailsContent = (
                                            <>
                                                {detailsContent}
                                                <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-all', fontSize: '0.75rem', marginTop: '4px' }}>
                                                    {typeof anomaly.details === 'string' ? anomaly.details : JSON.stringify(anomaly.details, null, 2)}
                                                </pre>
                                            </>
                                         )
                                    }
                                }

                                return (
                                    <React.Fragment key={anomaly.id}>
                                        <ListItem 
                                            button 
                                            onClick={() => handleToggleAnomaly(anomaly.id)} 
                                            sx={{
                                                borderBottom: '1px solid', 
                                                borderColor: 'divider', 
                                                alignItems: 'center', // Center vertically
                                                py: 1.5
                                            }}
                                        >
                                            {/* Убираем отдельную иконку - она теперь в Chip */}
                                            {/* <ListItemIcon sx={{ mt: '6px' }}> */}
                                                {/* {severityIcon} */}
                                            {/* </ListItemIcon> */}
                                            <ListItemText
                                                primary={primaryText} // Теперь это Stack с Chip и текстом
                                                // Убираем secondary, т.к. время уже в primary
                                                // secondary={...} 
                                            />
                                            <Box sx={{ display: 'flex', alignItems: 'center', ml: 1 }}>
                                                {/* Добавляем Score */} 
                                                <Chip 
                                                    label={`Score: ${anomaly.anomaly_score?.toFixed(3) ?? 'N/A'}`}
                                                    size="small"
                                                    variant="outlined"
                                                    sx={{ height: 20, fontSize: '0.7rem' }}
                                                />
                                                <IconButton size="small" sx={{ ml: 0.5 }}>
                                                    {openAnomalyId === anomaly.id ? <ExpandLess /> : <ExpandMore />}
                                                </IconButton>
                                            </Box>
                                        </ListItem>
                                        <Collapse in={openAnomalyId === anomaly.id} timeout="auto" unmountOnExit>
                                            <Box sx={{ pl: 4, pr: 2, py: 2, bgcolor: (theme) => alpha(theme.palette.grey[100], 0.5) }}>
                                                <Typography variant="body2" gutterBottom sx={{ fontWeight: 'bold' }}>Детали:</Typography>
                                                {detailsContent} {/* Отображаем отформатированные детали */}
                                            </Box>
                                        </Collapse>
                                    </React.Fragment>
                                );
                            })}
                        </List>
                    )}
                </Paper>
            </Grid>
        </Grid>
    );
}

export default DashboardPage; 
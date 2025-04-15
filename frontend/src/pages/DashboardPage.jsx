import React, { useState, useEffect, useCallback, useMemo } from 'react';
import axios from 'axios';
import {
    Typography, Paper, Grid, CircularProgress, Alert, Card, CardContent, Button,
    Box, List, ListItem, ListItemText, Divider, Chip, Stack, useTheme, IconButton, ListItemIcon, Collapse,
    Dialog, DialogTitle, DialogContent, DialogActions, TableContainer, Table, TableBody, TableRow, TableCell, Link as MuiLink,
    Accordion, AccordionSummary, AccordionDetails, Select, MenuItem, FormControl, InputLabel, TableHead
} from '@mui/material';
import { 
    Train, Search, ExpandLess, ExpandMore, ErrorOutline, CheckCircleOutline, Settings, WarningAmberOutlined, InfoOutlined,
    Close as CloseIcon, Link as LinkIcon, BugReport, HistoryToggleOff,
    ExpandMore as ExpandMoreIcon, Dashboard as DashboardIcon
} from '@mui/icons-material';
import { useNavigate, Link as RouterLink } from 'react-router-dom';
import { format, parseISO } from 'date-fns';
import { ru } from 'date-fns/locale';
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
    Filler
} from 'chart.js';
import { Line, Bar } from 'react-chartjs-2'; // Импортируем типы графиков
import { alpha } from '@mui/material/styles';
// --- ИМПОРТИРУЕМ ZOOM ПЛАГИН (если еще не импортирован глобально) ---
import zoomPlugin from 'chartjs-plugin-zoom';
import 'hammerjs'; // Для поддержки touch-жестов
// -----------------------------------------------------------------

// Регистрируем компоненты Chart.js
ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    BarElement,
    Title,
    Tooltip,
    Legend,
    Filler,
    zoomPlugin // <--- Добавляем регистрацию zoomPlugin
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

// --- Обновляем функцию чтения настроек --- 
const getAnomalySettings = () => {
    const savedSettings = localStorage.getItem('anomalySettings');
    // Добавляем дефолты для автоэнкодера
    const defaults = {
        limit: 10000,
        z_threshold: 3.0,
        dbscan_eps: 0.5,
        dbscan_min_samples: 5,
        use_autoencoder: false, // <-- Новый дефолт
        autoencoder_threshold: 0.5, // <-- Новый дефолт
        use_isolation_forest: true, // <-- Добавляем дефолт для IF
        use_dbscan: true // <-- Добавляем дефолт для DBSCAN (включен)
    };
    if (savedSettings) {
        try {
            const parsed = JSON.parse(savedSettings);
            // Важно: Объединяем сохраненные с дефолтными, чтобы новые ключи подхватились
            return {
                ...defaults, // Начинаем с дефолтов
                // Перезаписываем сохраненными, если они есть и валидны
                limit: parseInt(parsed.limit, 10) || defaults.limit,
                z_threshold: parseFloat(parsed.z_threshold) || defaults.z_threshold,
                dbscan_eps: parseFloat(parsed.dbscan_eps) || defaults.dbscan_eps,
                dbscan_min_samples: parseInt(parsed.dbscan_min_samples, 10) || defaults.dbscan_min_samples,
                use_autoencoder: typeof parsed.use_autoencoder === 'boolean' ? parsed.use_autoencoder : defaults.use_autoencoder,
                autoencoder_threshold: parseFloat(parsed.autoencoder_threshold) || defaults.autoencoder_threshold,
                // --- Читаем use_isolation_forest --- 
                use_isolation_forest: typeof parsed.use_isolation_forest === 'boolean' ? parsed.use_isolation_forest : defaults.use_isolation_forest,
                // --- Читаем use_dbscan --- 
                use_dbscan: typeof parsed.use_dbscan === 'boolean' ? parsed.use_dbscan : defaults.use_dbscan,
                // ---------------------------------
            };
        } catch (error) {
            console.error("Failed to parse settings from localStorage:", error);
            return defaults;
        }
    }
    return defaults;
};
// -----------------------------------------

// --- Функция для получения цвета чипа серьезности ---
const getSeverityChipColor = (severity) => {
    const upperSeverity = severity?.toUpperCase();
    if (upperSeverity === 'HIGH') return 'error';
    if (upperSeverity === 'MEDIUM') return 'warning';
    if (upperSeverity === 'LOW') return 'info';
    return 'default';
};

// --- Компонент для отображения деталей аномалии в модальном окне ---
// ТЕПЕРЬ принимает ConsolidatedAnomaly
const AnomalyDetailsDialogContent = ({ anomaly }) => {
    const theme = useTheme();

    // --- Состояние для истории заказов --- 
    const [customerOrderHistory, setCustomerOrderHistory] = useState([]);
    const [loadingOrderHistory, setLoadingOrderHistory] = useState(false);
    const [errorOrderHistory, setErrorOrderHistory] = useState('');
    
    // --- НОВОЕ: Состояние для истории сессии --- 
    const [sessionActivityHistory, setSessionActivityHistory] = useState([]);
    const [loadingSessionHistory, setLoadingSessionHistory] = useState(false);
    const [errorSessionHistory, setErrorSessionHistory] = useState('');
    // --------------------------------------------

    // --- Обновленный useEffect для загрузки контекста ---
    useEffect(() => {
        // Сбрасываем все состояния перед загрузкой
        setCustomerOrderHistory([]); setLoadingOrderHistory(false); setErrorOrderHistory('');
        setSessionActivityHistory([]); setLoadingSessionHistory(false); setErrorSessionHistory('');

        if (anomaly) {
            if (anomaly.entity_type === 'order') {
                // Загрузка истории заказов (как было)
                const fetchOrderHistory = async () => {
                    setLoadingOrderHistory(true);
                    try {
                        const response = await axios.get(`/api/order_data/orders/${anomaly.entity_id}/context/customer_history`, {
                             params: { limit: 7 }
                        });
                        setCustomerOrderHistory(response.data);
                    } catch (err) {
                        console.error("Error fetching customer order history:", err);
                        setErrorOrderHistory(err.response?.data?.detail || 'Не удалось загрузить историю заказов клиента.');
                    }
                     finally { setLoadingOrderHistory(false); }
                };
                fetchOrderHistory();

            } else if (anomaly.entity_type === 'activity') {
                 // --- НОВОЕ: Загрузка истории сессии --- 
                 const fetchSessionHistory = async () => {
                    setLoadingSessionHistory(true);
                    try {
                        const response = await axios.get(`/api/activities/${anomaly.entity_id}/context/session_history`);
                        setSessionActivityHistory(response.data); // Ожидаем List[SimpleActivityHistoryItem]
                    } catch (err) {
                        console.error("Error fetching session activity history:", err);
                        setErrorSessionHistory(err.response?.data?.detail || 'Не удалось загрузить историю сессии.');
                    }
                    finally { setLoadingSessionHistory(false); }
                 };
                 fetchSessionHistory();
                 // -------------------------------------
            }
        }
    }, [anomaly]); 
    // ---------------------------------------------------------

    if (!anomaly) return null;

    // Форматируем дату последнего обнаружения
    let lastDetectedFormatted = 'N/A';
    if (anomaly.last_detected_at) {
        try {
            // Даты из FastAPI обычно в ISO формате
            lastDetectedFormatted = format(parseISO(anomaly.last_detected_at), 'dd.MM.yyyy HH:mm:ss', { locale: ru });
        } catch (e) { console.error("Date format error:", e); }
    }
    
    // Получаем цвет для общей серьезности
    const overallSeverityColor = getSeverityChipColor(anomaly.overall_severity);

    return (
        <Box>
            {/* --- Сводная информация --- */}
            <Typography variant="subtitle1" gutterBottom>Общая информация</Typography>
            <TableContainer component={Paper} elevation={0} variant="outlined" sx={{ mb: 2 }}>
                <Table size="small">
                    <TableBody>
                        <TableRow><TableCell sx={{fontWeight: 'bold'}}>Тип Сущности:</TableCell><TableCell>{anomaly.entity_type}</TableCell></TableRow>
                        <TableRow><TableCell sx={{fontWeight: 'bold'}}>ID Сущности:</TableCell><TableCell>{anomaly.entity_id}</TableCell></TableRow>
                        <TableRow><TableCell sx={{fontWeight: 'bold'}}>Общая Серьезность:</TableCell><TableCell>
                             <Chip 
                                label={(anomaly.overall_severity || 'Unknown').toUpperCase()}
                                color={overallSeverityColor}
                                size="small"
                            />
                        </TableCell></TableRow>
                        <TableRow><TableCell sx={{fontWeight: 'bold'}}>Кол-во Детекторов:</TableCell><TableCell>{anomaly.detector_count}</TableCell></TableRow>
                        <TableRow><TableCell sx={{fontWeight: 'bold'}}>Время Обнаружения:</TableCell><TableCell>{lastDetectedFormatted}</TableCell></TableRow>
                        {/* Добавляем ссылку на страницу сущности */}
                        {(anomaly.entity_type === 'activity' || anomaly.entity_type === 'order') && (
                             <TableRow>
                                <TableCell colSpan={2} align="right">
                                     <MuiLink component={RouterLink} 
                                            to={`/${anomaly.entity_type}s?highlight=${anomaly.entity_id}`} 
                                            underline="hover" 
                                            target="_blank" // Открывать в новой вкладке
                                            rel="noopener noreferrer"
                                        > 
                                         <LinkIcon fontSize="small" sx={{ verticalAlign: 'bottom', mr: 0.5 }} /> 
                                         Перейти к {anomaly.entity_type === 'activity' ? 'активности' : 'заказу'}
                                     </MuiLink>
                                </TableCell>
                             </TableRow>
                        )}
                    </TableBody>
                </Table>
            </TableContainer>

            {/* --- НОВОЕ: Секция Истории Заказов Клиента --- */}
            {anomaly.entity_type === 'order' && (
                <Box sx={{ mt: 3 }}>
                    <Typography variant="subtitle1" gutterBottom>Недавние заказы этого клиента</Typography>
                    {loadingOrderHistory && <CircularProgress size={20} sx={{ display: 'block', mx: 'auto'}}/>}
                    {errorOrderHistory && <Alert severity="warning" size="small" sx={{mt: 1}}>{errorOrderHistory}</Alert>}
                    {!loadingOrderHistory && !errorOrderHistory && customerOrderHistory.length > 0 && (
                         <TableContainer component={Paper} elevation={0} variant="outlined" sx={{ mt: 1 }}>
                            <Table size="small" aria-label="customer order history">
                                <TableHead>
                                     <TableRow>
                                        <TableCell>ID Заказа</TableCell>
                                        <TableCell>Дата</TableCell>
                                        <TableCell align="right">Сумма</TableCell>
                                        <TableCell align="right">Кол-во поз.</TableCell>
                                     </TableRow>
                                </TableHead>
                                <TableBody>
                                    {customerOrderHistory.map((order) => (
                                        <TableRow 
                                            key={order.id}
                                            sx={{
                                                // Выделяем строку текущей аномалии
                                                backgroundColor: order.is_current_anomaly 
                                                    ? alpha(theme.palette.warning.light, 0.3) 
                                                    : 'inherit',
                                                '&:last-child td, &:last-child th': { border: 0 }
                                            }}
                                        >
                                            <TableCell component="th" scope="row">
                                                {order.id}
                                                {order.is_current_anomaly && <Chip label="Текущая" size="small" color="warning" variant="outlined" sx={{ml: 1, height: 18}}/>}
                                            </TableCell>
                                            <TableCell>{format(parseISO(order.created_at), 'dd.MM.yy HH:mm', { locale: ru })}</TableCell>
                                            <TableCell align="right">{order.total_amount?.toFixed(2) ?? '-'}</TableCell>
                                            <TableCell align="right">{order.item_count}</TableCell>
                                        </TableRow>
                                    ))}
                                </TableBody>
                            </Table>
                        </TableContainer>
                    )}
                    {!loadingOrderHistory && !errorOrderHistory && customerOrderHistory.length === 0 && anomaly.entity_type === 'order' && (
                        <Typography variant="body2" color="text.secondary" sx={{mt: 1}}>Нет данных о других заказах этого клиента.</Typography>
                    )}
                </Box>
            )}
            {/* ------------------------------------------------- */}

            {/* --- НОВОЕ: Секция Истории Сессии --- */}
            {anomaly.entity_type === 'activity' && (
                <Box sx={{ mt: 3 }}>
                    <Typography variant="subtitle1" gutterBottom>Действия в рамках этой сессии</Typography>
                    {loadingSessionHistory && <CircularProgress size={20} sx={{ display: 'block', mx: 'auto'}}/>}
                    {errorSessionHistory && <Alert severity="warning" size="small" sx={{mt: 1}}>{errorSessionHistory}</Alert>}
                    {!loadingSessionHistory && !errorSessionHistory && sessionActivityHistory.length > 0 && (
                         <TableContainer component={Paper} elevation={0} variant="outlined" sx={{ mt: 1, maxHeight: 300, overflowY: 'auto' }}>
                            <Table size="small" aria-label="session activity history" stickyHeader>
                                <TableHead>
                                     <TableRow>
                                        <TableCell sx={{width: '20%'}}>Время</TableCell>
                                        <TableCell sx={{width: '25%'}}>Тип Действия</TableCell>
                                        <TableCell>Детали</TableCell>
                                     </TableRow>
                                </TableHead>
                                <TableBody>
                                    {sessionActivityHistory.map((activity) => (
                                        <TableRow 
                                            key={activity.id}
                                            sx={{
                                                backgroundColor: activity.is_current_anomaly 
                                                    ? alpha(theme.palette.warning.light, 0.3) 
                                                    : 'inherit',
                                                '&:last-child td, &:last-child th': { border: 0 }
                                            }}
                                        >
                                            <TableCell>
                                                 {format(parseISO(activity.timestamp), 'HH:mm:ss.SSS', { locale: ru })}
                                                 {activity.is_current_anomaly && <Chip label="Текущая" size="small" color="warning" variant="outlined" sx={{ml: 1, height: 18}}/>}
                                            </TableCell>
                                            <TableCell>{activity.action_type}</TableCell>
                                            <TableCell sx={{fontSize: '0.75rem'}}>
                                                 {activity.details ? JSON.stringify(activity.details) : '-'} {/* Просто выводим JSON деталей */} 
                                            </TableCell>
                                        </TableRow>
                                    ))}
                                </TableBody>
                            </Table>
                        </TableContainer>
                    )}
                    {!loadingSessionHistory && !errorSessionHistory && sessionActivityHistory.length === 0 && anomaly.entity_type === 'activity' && (
                        <Typography variant="body2" color="text.secondary" sx={{mt: 1}}>Нет данных о других действиях в этой сессии.</Typography>
                    )}
                     {/* Добавляем session_id, если он есть */} 
                     {anomaly.entity_type === 'activity' && sessionActivityHistory.length > 0 && (
                         <Typography variant="caption" display="block" color="text.secondary" sx={{mt: 0.5}}>
                             Session ID: {sessionActivityHistory[0]?.session_id || 'N/A'} {/* Предполагаем, что session_id одинаковый для всех */} 
                         </Typography>
                     )}
                </Box>
            )}
            {/* ------------------------------------------ */}

            {/* --- Детали по каждому детектору (используем аккордеон) --- */}
            <Typography variant="subtitle1" gutterBottom sx={{ mt: 3 }}>Сработавшие Детекторы</Typography>
            {anomaly.triggered_detectors && anomaly.triggered_detectors.length > 0 ? (
                anomaly.triggered_detectors
                    // Сортируем для консистентности, например, по имени
                    .sort((a, b) => a.detector_name.localeCompare(b.detector_name))
                    .map((detectorInfo, index) => {
                        let detectorTimeFormatted = 'N/A';
                        if (detectorInfo.detection_timestamp) {
                             try {
                                detectorTimeFormatted = format(parseISO(detectorInfo.detection_timestamp), 'HH:mm:ss.SSS', { locale: ru });
                             } catch {} 
                        }
                        const severityColor = getSeverityChipColor(detectorInfo.severity);

                         // Форматируем details
                         let detailsContent = null;
                         if (detectorInfo.details !== null && detectorInfo.details !== undefined) {
                            if (typeof detectorInfo.details === 'object') {
                                // --- ДОБАВЛЯЕМ ЛОГИКУ ИЗВЛЕЧЕНИЯ feature_errors ---
                                let featureErrors = null;
                                let otherDetails = { ...detectorInfo.details }; // Копируем детали

                                if (detectorInfo.detector_name === 'autoencoder' && typeof detectorInfo.details === 'object' && detectorInfo.details !== null && detectorInfo.details.feature_errors) {
                                    featureErrors = { ...detectorInfo.details.feature_errors };
                                    if (featureErrors) {
                                        delete otherDetails.feature_errors; // Удаляем ошибки из "основных" деталей
                                    }
                                }
                                // ----------------------------------------------

                                detailsContent = (
                                     <Box>
                                         {/* Отображаем остальные детали */} 
                                         {Object.keys(otherDetails).length > 0 && (
                                             <TableContainer component={Paper} elevation={0} variant="outlined" sx={{ mt: 1, mb: featureErrors ? 2 : 0 }}>
                                        <Table size="small">
                                            <TableBody>
                                                        {Object.entries(otherDetails).map(([key, value]) => (
                                                    <TableRow key={key}>
                                                                <TableCell sx={{fontWeight: 'bold', width: '40%', py: 0.5 }}>{String(key).replace(/_/g, ' ')}:</TableCell> {/* Улучшаем читаемость ключа */} 
                                                                <TableCell sx={{ py: 0.5 }}>{typeof value === 'number' ? value.toFixed(4) : String(value)}</TableCell> {/* Форматируем числа */} 
                                                    </TableRow>
                                                ))}
                                            </TableBody>
                                        </Table>
                                    </TableContainer>
                                         )}

                                        {/* --- ЗАМЕНЯЕМ ТАБЛИЦУ MAE НА Bar Chart --- */}
                                        {featureErrors && (
                                             <>
                                                <Typography variant="subtitle2" gutterBottom sx={{ mt: 1.5, color: 'text.secondary' }}>
                                                     Ошибки реконструкции по признакам (MAE):
                                                </Typography>
                                                {/* Подготовка данных и рендеринг графика MAE */}
                                                {(() => {
                                                    const maeEntries = Object.entries(featureErrors)
                                                        .sort(([, errorA], [, errorB]) => errorB - errorA) // Сортируем по убыванию MAE
                                                        .slice(0, 15); // Показываем топ-15 признаков

                                                    const chartLabels = maeEntries.map(([feature]) => feature);
                                                    const chartDataValues = maeEntries.map(([, error]) => error);

                                                    // Используем один цвет (например, красный), т.к. MAE всегда >= 0
                                                    const backgroundColors = chartDataValues.map(() => 'rgba(255, 99, 132, 0.6)');
                                                    const borderColors = chartDataValues.map(() => 'rgba(255, 99, 132, 1)');

                                                    const chartData = {
                                                        labels: chartLabels,
                                                        datasets: [
                                                            {
                                                                label: 'MAE',
                                                                data: chartDataValues,
                                                                backgroundColor: backgroundColors,
                                                                borderColor: borderColors,
                                                                borderWidth: 1,
                                                            },
                                                        ],
                                                    };

                                                    const chartOptions = {
                                                        indexAxis: 'y', // Горизонтальный бар
                                                        responsive: true,
                                                        maintainAspectRatio: false,
                                                        plugins: {
                                                            legend: { display: false },
                                                            tooltip: {
                                                                callbacks: {
                                                                    label: function(context) {
                                                                        return ` ${context.dataset.label}: ${context.raw.toFixed(4)}`;
                                                                    }
                                                                }
                                                            }
                                                        },
                                                        scales: {
                                                            x: {
                                                                title: {
                                                                    display: true,
                                                                    text: 'MAE по признаку',
                                                                    font: { size: 10 }
                                                                },
                                                                ticks: { font: { size: 9 } }
                                                            },
                                                            y: {
                                                                 ticks: { font: { size: 9 } }
                                                            }
                                                        }
                                                    };

                                                    return (
                                                        <Box sx={{ height: '300px', mt: 1 }}> {/* Задаем высоту контейнера графика */}
                                                            <Bar options={chartOptions} data={chartData} />
                                                        </Box>
                                                    );
                                                })()}
                                            </>
                                        )}
                                        {/* --- График SHAP для Isolation Forest (остается без изменений) --- */}
                                        {detectorInfo.detector_name === 'isolation_forest' && detectorInfo.details?.shap_values && (
                                             <>
                                                <Typography variant="subtitle2" gutterBottom sx={{ mt: 1.5, color: 'text.secondary' }}>
                                                     Вклад признаков (SHAP values):
                                                </Typography>
                                                {/* Подготовка данных для графика SHAP */}
                                                {(() => {
                                                    const shapEntries = Object.entries(detectorInfo.details.shap_values)
                                                        .sort(([, valA], [, valB]) => Math.abs(valB) - Math.abs(valA))
                                                        .slice(0, 15); // Показываем топ-15 признаков
                                                    
                                                    const chartLabels = shapEntries.map(([feature]) => feature);
                                                    const chartDataValues = shapEntries.map(([, value]) => value);
                                                    const backgroundColors = chartDataValues.map(value => 
                                                        value < 0 ? 'rgba(255, 99, 132, 0.6)' : 'rgba(75, 192, 192, 0.6)' // Красный для отриц., Бирюзовый для полож.
                                                    );
                                                    const borderColors = chartDataValues.map(value => 
                                                        value < 0 ? 'rgba(255, 99, 132, 1)' : 'rgba(75, 192, 192, 1)'
                                                    );

                                                    const chartData = {
                                                        labels: chartLabels,
                                                        datasets: [
                                                            {
                                                                label: 'SHAP Value',
                                                                data: chartDataValues,
                                                                backgroundColor: backgroundColors,
                                                                borderColor: borderColors,
                                                                borderWidth: 1,
                                                            },
                                                        ],
                                                    };

                                                    const chartOptions = {
                                                        indexAxis: 'y', // Горизонтальный бар
                                                        responsive: true,
                                                        maintainAspectRatio: false,
                                                        plugins: {
                                                            legend: {
                                                                display: false, // Легенда не нужна
                                                            },
                                                            tooltip: {
                                                                callbacks: {
                                                                    label: function(context) {
                                                                        return ` ${context.dataset.label}: ${context.raw.toFixed(4)}`;
                                                                    }
                                                                }
                                                            }
                                                        },
                                                        scales: {
                                                            x: {
                                                                title: {
                                                                    display: true,
                                                                    text: 'SHAP Value (влияние на оценку аномальности)',
                                                                    font: { size: 10 }
                                                                },
                                                                ticks: { font: { size: 9 } }
                                                            },
                                                            y: {
                                                                 ticks: { font: { size: 9 } } 
                                                            }
                                                        }
                                                    };

                                                    return (
                                                        <Box sx={{ height: '300px', mt: 1 }}> {/* Задаем высоту контейнера графика */} 
                                                            <Bar options={chartOptions} data={chartData} />
                                                        </Box>
                                                    );
                                                })()}
                                            </>
                                        )}
                                        {/* --------------------------------------------- */}
                                    </Box>
                                );
                            } else { // Если details не объект
                                detailsContent = (
                                    <Paper elevation={0} variant="outlined" sx={{ p: 1, mt: 1, bgcolor: 'grey.50' }}>
                                        <pre style={{ margin: 0, fontSize: '0.75rem', whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
                                            {String(detectorInfo.details)}
                                        </pre>
                                    </Paper>
                                );
                            }
                        }

                        return (
                            <Accordion key={index} elevation={1} sx={{ mb: 1 }}>
                                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                                    <Stack direction="row" spacing={1.5} alignItems="center" sx={{ width: '100%' }}>
                                        <Chip label={detectorInfo.detector_name} size="small" variant="outlined" />
                                        <Chip label={detectorInfo.severity || 'N/A'} size="small" color={severityColor} />
                                        {detectorInfo.anomaly_score !== null && (
                                             <Chip label={`Score: ${detectorInfo.anomaly_score.toFixed(3)}`} size="small" variant="outlined" />
                                        )}
                                        <Box sx={{ flexGrow: 1 }} /> {/* Занимает оставшееся место */} 
                                        <Typography variant="caption" color="text.secondary">{detectorTimeFormatted}</Typography>
                                    </Stack>
                                </AccordionSummary>
                                <AccordionDetails sx={{ bgcolor: 'action.hover' }}>
                                     <Typography variant="subtitle2" gutterBottom sx={{ color: 'text.secondary', mt: 1 }}>
                                        Объяснение:
                                     </Typography>
                                     <Typography 
                                        variant="body2" 
                                        paragraph 
                                        sx={{ 
                                            fontStyle: 'italic', 
                                            mb: 2, 
                                            pl: 1.5, // Отступ слева
                                            borderLeft: '3px solid', 
                                            borderColor: 'divider' 
                                        }}
                                    >
                                        {detectorInfo.reason || 'Объяснение не предоставлено.'} 
                                    </Typography>
                                    <Typography variant="subtitle2" gutterBottom>Детали Обнаружения:</Typography>
                                    {detailsContent ? detailsContent : <Typography variant="body2" color="text.secondary">Нет деталей.</Typography>}
                                </AccordionDetails>
                            </Accordion>
                        );
                    })
            ) : (
                <Typography color="text.secondary">Нет информации по детекторам.</Typography>
            )}
        </Box>
    );
};

// --- Anomaly Details Modal Component ---
const AnomalyDetailsModal = ({ open, onClose, anomalyDetails, loading, error }) => {
    // ... (imports, helper functions like getSeverityColor, formatTimestamp - no changes) ...

    return (
        <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
            {/* ... (DialogTitle, DialogContent start, Loading/Error handling) ... */}
                {anomalyDetails && (
                    <Box>
                        {/* ... (Общая информация section - no changes) ... */}

                        <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>Сработавшие Детекторы</Typography>
                        {anomalyDetails.detected_by && anomalyDetails.detected_by.map((detectorInfo, index) => (
                            <Accordion key={index} defaultExpanded={index === 0}>
                                <AccordionSummary
                                    expandIcon={<ExpandMoreIcon />}
                                    aria-controls={`panel${index}a-content`}
                                    id={`panel${index}a-header`}
                                >
                                    {/* ... (AccordionSummary content - detector name, severity, timestamp - no changes) ... */}
                                </AccordionSummary>
                                <AccordionDetails>
                                    {/* --- ДОБАВЛЕНО: Отображение Reason --- */} 
                                    <Typography variant="subtitle2" gutterBottom sx={{ color: 'text.secondary', mt: 1 }}> 
                                        Объяснение:
                                    </Typography>
                                    <Typography 
                                        variant="body2" 
                                        paragraph 
                                        sx={{ 
                                            fontStyle: 'italic', 
                                            mb: 2, 
                                            pl: 1, 
                                            borderLeft: '3px solid', 
                                            borderColor: 'divider' 
                                        }}
                                    >
                                        {detectorInfo.reason || 'Объяснение не предоставлено.'} 
                                    </Typography>
                                    {/* ------------------------------------- */} 

                                    <Typography variant="subtitle2" gutterBottom sx={{ color: 'text.secondary' }}>
                                        Детали Обнаружения:
                                    </Typography>
                                    <Paper elevation={0} variant="outlined" sx={{ p: 1.5, bgcolor: 'grey.50' }}>
                                        {/* ... (Mapping through detectorInfo.details - no changes) ... */}
                                        {!detectorInfo.details && <Typography variant="body2">Детали отсутствуют.</Typography>}
                                    </Paper>
                                </AccordionDetails>
                            </Accordion>
                        ))}
                        {/* ... (Handling no detectors) ... */}
                    </Box>
                )}
            {/* ... (DialogContent end, DialogActions) ... */}
        </Dialog>
    );
};

// --- Импорт нового компонента графика --- 
import ScoreDistributionChart from '../components/ScoreDistributionChart';
// --- ДОБАВЛЯЕМ Pagination --- 
import Pagination from '@mui/material/Pagination';
// -------------------------

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
    // --- СОСТОЯНИЕ ДЛЯ ПАГИНАЦИИ АНОМАЛИЙ --- 
    const [currentPage, setCurrentPage] = useState(1);
    const [anomaliesPerPage, setAnomaliesPerPage] = useState(15); // Кол-во аномалий на странице
    const [totalAnomalies, setTotalAnomalies] = useState(0);
    // ----------------------------------------

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

    // --- NEW State for Anomaly Detail Modal ---
    const [selectedAnomaly, setSelectedAnomaly] = useState(null);
    const [isDetailModalOpen, setIsDetailModalOpen] = useState(false);
    // -----------------------------------------

    // --- State for Modal ---
    const [modalOpen, setModalOpen] = useState(false);
    const [selectedAnomalyDetails, setSelectedAnomalyDetails] = useState(null);
    const [modalLoading, setModalLoading] = useState(false);
    const [modalError, setModalError] = useState('');

    // --- Состояние для графика распределения скоров --- 
    const [selectedDetector, setSelectedDetector] = useState('isolation_forest'); // Детектор по умолчанию
    const [scoreData, setScoreData] = useState([]);
    const [scoreLoading, setScoreLoading] = useState(false);
    const [scoreError, setScoreError] = useState('');
    // --------------------------------------------------

    // --- Получаем актуальные настройки для порогов ---
    const currentSettings = useMemo(() => getAnomalySettings(), []); // Получаем настройки один раз

    // --- Определяем порог для графика скоров ---
    const currentThreshold = useMemo(() => {
        if (selectedDetector === 'statistical_zscore') {
            return currentSettings.z_threshold;
        } else if (selectedDetector === 'autoencoder') {
            return currentSettings.autoencoder_threshold;
        }
        return null; // Для других детекторов порог не показываем
    }, [selectedDetector, currentSettings]);
    // -----------------------------------------

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
                    borderColor: 'rgb(53, 162, 235)', // Синий цвет линии
                    backgroundColor: 'rgba(53, 162, 235, 0.5)', // Синий цвет заливки с 50% прозрачности
                    fill: true, // Включаем заливку под линией
                    tension: 0.1,
                },
            ],
        };
    }, [activityTimelineData]);

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
            tooltip: {
                callbacks: {
                    title: function(tooltipItems) {
                        // Показываем дату/интервал в заголовке тултипа
                        return `Дата: ${tooltipItems[0].label}`;
                    },
                    label: function(context) {
                        // Показываем значение в основной строке
                        return ` Активность: ${context.parsed.y}`;
                    }
                }
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

    // --- Функция для ЗАГРУЗКИ КОНСОЛИДИРОВАННЫХ аномалий --- 
    // --- ОБНОВЛЯЕМ для пагинации --- 
    const fetchAnomalies = useCallback(async (page = 1) => {
        setLoadingAnomalies(true);
        setErrorAnomalies('');
        // Не очищаем anomalies, чтобы не было мерцания при переключении страниц?
        // Или очищать? Пока не будем: // setAnomalies([]); 
        
        const limit = anomaliesPerPage;
        const skip = (page - 1) * limit;

        try {
            // Используем эндпоинт, который теперь возвращает { total_count, anomalies }
            const response = await axios.get('/api/anomalies/', { params: { skip: skip, limit: limit } });
            // Ожидаем объект ConsolidatedAnomalyResponse
            if (response?.data && Array.isArray(response.data.anomalies) && typeof response.data.total_count === 'number') {
                setAnomalies(response.data.anomalies);
                setTotalAnomalies(response.data.total_count);
                setCurrentPage(page); // Устанавливаем текущую страницу
            } else {
                 console.error("Received unexpected data format for paginated anomalies", response?.data);
                 setErrorAnomalies("Некорректный формат данных аномалий.");
                 setAnomalies([]); // Очищаем в случае ошибки формата
                 setTotalAnomalies(0);
            }
        } catch (err) {
            console.error("Error fetching anomalies:", err);
            setErrorAnomalies(err.response?.data?.detail || 'Не удалось загрузить список аномалий.');
            setAnomalies([]); // Очищаем при ошибке загрузки
            setTotalAnomalies(0);
        } finally {
            setLoadingAnomalies(false);
        }
    }, [anomaliesPerPage]); // Добавляем зависимость
    // -------------------------------------------------------------------

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

    // --- Функция для загрузки скоров --- 
    const fetchScores = useCallback(async (detector) => {
        if (!detector) return;
        setScoreLoading(true);
        setScoreError('');
        setScoreData([]); // Очищаем предыдущие данные
        try {
            const response = await axios.get(`/api/charts/anomaly_scores?detector_name=${detector}`);
            if (Array.isArray(response.data)) {
                setScoreData(response.data);
            } else {
                console.error("Unexpected score data format:", response.data);
                setScoreError('Неверный формат данных скоров.');
            }
        } catch (err) { 
            console.error(`Error fetching scores for ${detector}:`, err);
            setScoreError(`Ошибка загрузки скоров для ${detector}. ${err.response?.data?.detail || err.message}`);
        } finally {
            setScoreLoading(false);
        }
    }, []);
    // ------------------------------------

    // Загрузка всех данных при монтировании
    useEffect(() => {
        fetchDashboardData();
        fetchAnomalies(1); // Загружаем первую страницу аномалий
        fetchActivityTimeline(); // Загружаем график активности (по дням по умолчанию)
        fetchAnomalySummary();   // Загружаем сводку аномалий
        // Загружаем скоры для детектора по умолчанию при монтировании
        fetchScores(selectedDetector);
    }, [fetchDashboardData, fetchAnomalies, fetchActivityTimeline, fetchAnomalySummary, fetchScores, selectedDetector]);

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

    // --- ОБНОВЛЕННЫЙ ОБРАБОТЧИК ДЕТЕКЦИИ (ACTIVITY) --- 
    const handleDetectActivityAnomalies = async () => {
        setLoadingDetectActivity(true);
        setDetectionResult(null);
        try {
            const currentSettings = getAnomalySettings();
            console.log('Current settings for Activity detection:', currentSettings);

            const algorithmsToRun = ['statistical_zscore', 'isolation_forest', 'dbscan'];
            if (currentSettings.use_autoencoder) {
                algorithmsToRun.push('autoencoder');
            }
            console.log('Algorithms to run for Activity:', algorithmsToRun);

            const response = await axios.post('/api/anomalies/detect', {
                entity_type: 'activity',
                algorithms: algorithmsToRun,
                limit: currentSettings.limit,
                z_threshold: currentSettings.z_threshold,
                dbscan_eps: currentSettings.dbscan_eps,
                dbscan_min_samples: currentSettings.dbscan_min_samples,
            });
            setDetectionResult({ success: true, data: response.data });
            fetchAnomalies();
        } catch (error) { 
            console.error("Error detecting activity anomalies:", error);
            const errorMsg = error.response?.data?.detail || error.message || 'Failed to run activity anomaly detection';
            setDetectionResult({ success: false, message: errorMsg });
        }
        finally {
        setLoadingDetectActivity(false);
        }
    };

    // --- Обработчик детекции (Order) --- 
    // --- ОБНОВЛЯЕМ, чтобы использовать настройки --- 
    const handleDetectOrderAnomalies = async () => {
        setLoadingDetectOrder(true); 
        setDetectionResult(null); 
        setDetectError('');     
        try {
            const currentSettings = getAnomalySettings();
            console.log('Current settings for Order detection:', currentSettings);

            // --- Собираем алгоритмы для заказа --- 
            const algorithmsToRun = ['statistical_zscore']; 
            if (currentSettings.use_isolation_forest) {
                algorithmsToRun.push('isolation_forest');
            }
            // --- Добавляем проверку DBSCAN --- 
            if (currentSettings.use_dbscan) { 
                 algorithmsToRun.push('dbscan');
            }
            // --------------------------------
            if (currentSettings.use_autoencoder) {
                algorithmsToRun.push('autoencoder');
            }
            const uniqueAlgos = [...new Set(algorithmsToRun)]; 
            console.log('Algorithms to run for Order:', uniqueAlgos);
            // -------------------------------------
            
            // --- Передаем все нужные параметры --- 
            const payload = {
                entity_type: 'order',
                algorithms: uniqueAlgos, 
                limit: currentSettings.limit,
                z_threshold: currentSettings.z_threshold,
                dbscan_eps: currentSettings.dbscan_eps, 
                dbscan_min_samples: currentSettings.dbscan_min_samples, 
                autoencoder_threshold: currentSettings.autoencoder_threshold 
            };
            // -------------------------------------

            const response = await axios.post('/api/anomalies/detect', payload);
            console.log("Order detection response:", response.data);
            setDetectionResult({ success: true, data: response.data });
            fetchAnomalies(currentPage); // Обновляем список аномалий, оставаясь на текущей странице
        } catch (error) { 
            console.error("Error detecting order anomalies:", error);
            const errorMsg = error.response?.data?.detail || error.message || 'Failed to run order anomaly detection';
            setDetectionResult({ success: false, message: errorMsg });
            setDetectError(errorMsg); 
        } finally {
            setLoadingDetectOrder(false); 
        }
    };

    // --- Обработчик клика по строке аномалии --- 
    // Теперь просто открывает модальное окно с уже загруженными данными
    const handleToggleAnomaly = (anomaly) => {
        setSelectedAnomaly(anomaly); // Сохраняем всю консолидированную аномалию
        setIsDetailModalOpen(true);
    };

    // --- Закрытие модального окна (без изменений) --- 
    const handleCloseDetailModal = () => {
        setIsDetailModalOpen(false);
        setSelectedAnomaly(null);
    };

    // Navigate to Settings page
    const goToSettings = () => {
        navigate('/settings');
    };

    // Combined loading state for disabling buttons
    const isDetecting = loadingDetectActivity || loadingDetectOrder;

    // --- Fetch Anomaly Details for Modal ---
    const fetchAnomalyDetails = useCallback(async (anomalyId) => {
        setModalLoading(true);
        setModalError('');
        setSelectedAnomalyDetails(null); // Clear previous details
        try {
            const response = await axios.get(`/api/anomalies/${anomalyId}`);
            console.log("Anomaly Details Response:", response.data); // Log fetched details
            setSelectedAnomalyDetails(response.data);
        } catch (error) { // Explicitly catch the error
            console.error("Error fetching anomaly details:", error);
            setModalError(error.response?.data?.detail || error.message || 'Failed to fetch anomaly details');
        }
        setModalLoading(false);
    }, []);

    // --- Modal Handlers ---
    const handleOpenModal = (anomalyId) => {
        fetchAnomalyDetails(anomalyId);
        setModalOpen(true);
    };

    const handleCloseModal = () => {
        setModalOpen(false);
        setSelectedAnomalyDetails(null); // Clear details on close
        setModalError('');
    };

    // --- DataGrid Click Handler ---
    const handleRowClick = (params) => {
        // params.row contains the data of the clicked row
        console.log("Clicked Row ID:", params.row.id);
        handleOpenModal(params.row.id); // Open modal with the anomaly ID
    };

    // --- Обработчик смены детектора --- 
    const handleDetectorChange = (event) => {
        const newDetector = event.target.value;
        setSelectedDetector(newDetector);
        // Загружаем данные для нового детектора (useEffect сделает это, т.к. selectedDetector изменился)
        // fetchScores(newDetector); // Можно и так, но через useEffect надежнее
    };
    // ------------------------------------

    // --- ОБРАБОТЧИК СМЕНЫ СТРАНИЦЫ ПАГИНАЦИИ --- 
    const handlePageChange = (event, newPage) => {
        fetchAnomalies(newPage);
    };
    // ---------------------------------------------

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
                                <Grid xs={12} sm={6} md={4} lg={2.4} key={key}> {/* Adjust lg for 5 items */}
                                    {/* Стиль как у карточки Заказа */}
                                    <Card elevation={1}> {/* Оставляем небольшую тень */}
                                        <CardContent sx={{ p: 1.5 }}> {/* Возвращаем чуть больше padding */}
                                            {/* Название сущности (метка) */}
                                            <Typography variant="body2" component="div" sx={{ color: 'text.secondary', textTransform: 'capitalize', mb: 0.5 }}> {/* Меньше шрифт, отступ снизу */}
                                                {key.replace(/_/g, ' ')}
                                            </Typography>
                                            {/* Значение (количество) */}
                                            <Typography variant="h5" component="div" sx={{ fontWeight: 'medium' }}> {/* Крупнее и жирнее */}
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
            <Grid xs={12} md={7}> {/* Занимает большую часть ширины */}
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
                                         title: { ...chartOptions.plugins.title, text: 'Динамика активности' },
                                         // --- ДОБАВЛЯЕМ/ИЗМЕНЯЕМ ТУЛТИП ДЛЯ LINE CHART --- 
                                         tooltip: {
                                             callbacks: {
                                                 title: function(tooltipItems) {
                                                     // Показываем дату/интервал в заголовке тултипа
                                                     return `Дата: ${tooltipItems[0].label}`;
                                                 },
                                                 label: function(context) {
                                                     // Показываем значение в основной строке
                                                     return ` Активность: ${context.parsed.y}`;
                                                 }
                                             }
                                         },
                                         // -------------------------------------------------
                                         zoom: {
                                             pan: {
                                                 enabled: true,
                                                 mode: 'x', // Для таймлайна панорамирование по X более логично
                                                 threshold: 5,
                                             },
                                             zoom: {
                                                 wheel: { enabled: true },
                                                 pinch: { enabled: true },
                                                 mode: 'x', // Масштабируем тоже по оси X
                                             },
                                             limits: {
                                                // x: {min: 'original', max: 'original'} // Ограничения, если нужны
                                             }
                                         }
                                    }
                                }}
                                data={activityTimelineChartData}
                            />
                        </Box>
                    )}
                </Paper>
            </Grid>
            <Grid xs={12} md={5}> {/* Занимает оставшуюся часть */}
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
                                         legend: { display: false }, // Легенда не нужна для одного датасета
                                         // --- ДОБАВЛЯЕМ/ИЗМЕНЯЕМ ТУЛТИП ДЛЯ BAR CHART --- 
                                         tooltip: {
                                            callbacks: {
                                                title: function(tooltipItems) {
                                                    // Показываем имя детектора в заголовке
                                                    return `Детектор: ${tooltipItems[0].label}`;
                                                },
                                                label: function(context) {
                                                    // Показываем количество аномалий
                                                    return ` Кол-во аномалий: ${context.parsed.x}`;
                                                }
                                            }
                                        }
                                        // ------------------------------------------------
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
            <Grid xs={12}>
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
                    {detectError && <Alert severity="error" sx={{ mb: 2 }}>{detectError}</Alert>}
                    {detectionResult && detectionResult.success && detectionResult.data && (
                        <Alert severity="success" sx={{ mb: 2 }}>
                            {detectionResult.data.message}
                            {detectionResult.data.anomalies_saved_by_algorithm && 
                             Object.keys(detectionResult.data.anomalies_saved_by_algorithm).length > 0 && (
                                <ul>
                                    {Object.entries(detectionResult.data.anomalies_saved_by_algorithm).map(([algo, count]) => (
                                        <li key={algo}>{`${algo}: сохранено ${count} аномалий`}</li>
                                    ))}
                                </ul>
                            )}
                        </Alert>
                    )}
                    {detectionResult && !detectionResult.success && (
                        <Alert severity="error" sx={{ mb: 2 }}>
                             {detectionResult.message || 'Произошла ошибка при обнаружении аномалий.'}
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
                            {anomalies.map((anomaly, index) => {
                                const overallSeverityColor = getSeverityChipColor(anomaly.overall_severity);
                                
                                let lastDetectedTime = 'N/A';
                                try {
                                    lastDetectedTime = format(parseISO(anomaly.last_detected_at), 'dd.MM HH:mm', { locale: ru });
                                } catch {}

                                let primaryText = (
                                    <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap"> {/* flexWrap для переноса */} 
                                        <Chip 
                                            label={(anomaly.overall_severity || '-').toUpperCase()}
                                            color={overallSeverityColor}
                                            size="small"
                                            sx={{ mr: 1 }}
                                        />
                                        <Typography variant="body2" component="span" sx={{ mr: 1, fontWeight: 'medium' }}>
                                            {anomaly.entity_type.toUpperCase()} ID: {anomaly.entity_id}
                                        </Typography>
                                        <Chip 
                                            icon={<BugReport fontSize="small" />} 
                                            label={`${anomaly.detector_count}`}
                                            size="small"
                                            variant="outlined"
                                            title="Количество сработавших детекторов"
                                            sx={{ mr: 1 }}
                                        />
                                        {/* Список детекторов */} 
                                        {anomaly.triggered_detectors.slice(0, 3).map(d => ( // Показываем первые 3
                                            <Chip key={d.detector_name} label={d.detector_name} size="small" variant='outlined' sx={{ height: 20, fontSize: '0.7rem', mr: 0.5 }} />
                                        ))}
                                        {anomaly.detector_count > 3 && <Chip label={`+${anomaly.detector_count - 3}`} size="small" sx={{ height: 20, fontSize: '0.7rem' }}/>}
                                    </Stack>
                                );

                                return (
                                    <ListItem 
                                        key={`${anomaly.entity_type}-${anomaly.entity_id}-${index}`} // Генерируем ключ
                                        onClick={() => handleToggleAnomaly(anomaly)} // Открывает окно с ConsolidatedAnomaly
                                        sx={{ borderBottom: '1px solid', borderColor: 'divider', alignItems: 'center', py: 1.5 }}
                                    >
                                        <ListItemText 
                                            primary={primaryText} 
                                            secondary={`Последнее обнаружение: ${lastDetectedTime}`} 
                                            secondaryTypographyProps={{ variant: 'caption', sx: { mt: 0.5 } }}
                                        />
                                        <IconButton size="small" sx={{ ml: 1 }} aria-label="view details">
                                            <Search fontSize="small" /> 
                                        </IconButton>
                                    </ListItem>
                                );
                            })}
                        </List>
                    )}
                    {/* --- ДОБАВЛЯЕМ КОМПОНЕНТ ПАГИНАЦИИ --- */}
                    {!loadingAnomalies && totalAnomalies > 0 && totalAnomalies > anomaliesPerPage && (
                         <Box sx={{ display: 'flex', justifyContent: 'center', mt: 3, pb: 2 }}>
                            <Pagination 
                                count={Math.ceil(totalAnomalies / anomaliesPerPage)} 
                                page={currentPage}
                                onChange={handlePageChange}
                                color="primary" 
                                showFirstButton 
                                showLastButton
                            />
                        </Box>
                    )}
                    {/* ------------------------------------ */}
                </Paper>
            </Grid>
            
            {/* --- ОБНОВЛЕННОЕ Модальное Окно Деталей --- */}
            <Dialog 
                open={isDetailModalOpen} 
                onClose={handleCloseDetailModal} 
                maxWidth="lg" 
                fullWidth
                aria-labelledby="anomaly-detail-title"
            >
                <DialogTitle id="anomaly-detail-title">
                    {/* Заголовок теперь содержит entity_type и entity_id */} 
                    Детали Аномалии: {selectedAnomaly?.entity_type?.toUpperCase()} ID {selectedAnomaly?.entity_id ?? ''}
                    <IconButton /* ... кнопка закрытия ... */ > <CloseIcon /> </IconButton>
                </DialogTitle>
                <DialogContent dividers>
                    {/* Используем новый компонент контента */} 
                    {selectedAnomaly && <AnomalyDetailsDialogContent anomaly={selectedAnomaly} />}
                </DialogContent>
                <DialogActions>
                    <Button onClick={handleCloseDetailModal}>Закрыть</Button>
                </DialogActions>
            </Dialog>
            {/* --------------------------------------------- */}
            
            {/* Anomaly Details Modal Render */} 
            <AnomalyDetailsModal 
                open={modalOpen} 
                onClose={handleCloseModal} 
                anomalyDetails={selectedAnomalyDetails} 
                loading={modalLoading} 
                error={modalError} 
            />
            
            {/* --- График Распределения Скоров Аномальности --- */}
            <Grid xs={12} md={6}>
                <Paper elevation={3} sx={{ p: 2, borderRadius: 2, height: '100%', display: 'flex', flexDirection: 'column' }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                        <Typography variant="h6" gutterBottom sx={{ fontWeight: 'medium' }}>
                            Распределение Оценок
                        </Typography>
                        <FormControl size="small" sx={{ minWidth: 200 }}>
                            <InputLabel id="detector-select-label">Детектор</InputLabel>
                            <Select
                                labelId="detector-select-label"
                                value={selectedDetector}
                                label="Детектор"
                                onChange={handleDetectorChange}
                            >
                                {/* Доступные детекторы для графика скоров */} 
                                <MenuItem value={"isolation_forest"}>Isolation Forest</MenuItem>
                                <MenuItem value={"autoencoder"}>Autoencoder</MenuItem>
                                <MenuItem value={"statistical_zscore"}>Statistical (Z-Score)</MenuItem>
                            </Select>
                        </FormControl>
                    </Box>
                    {/* Вставляем компонент гистограммы */}
                    <ScoreDistributionChart 
                        scores={scoreData} 
                        detectorName={selectedDetector} 
                        isLoading={scoreLoading} 
                        error={scoreError} 
                        thresholdValue={currentThreshold}
                    />
                </Paper>
            </Grid>
            {/* ----------------------------------------------- */}
            
        </Grid>
    );
}

export default DashboardPage; 
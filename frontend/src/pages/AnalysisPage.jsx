import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import {
    Container, Typography, Paper, Grid, CircularProgress, Alert, Box, FormControl, InputLabel, Select, MenuItem, Button
} from '@mui/material';
import FeatureScatterPlot from '../components/FeatureScatterPlot'; // Импортируем будущий компонент

// --- Список доступных признаков для выбора --- 
const featureOptions = {
    order: [
        { value: 'total_amount', label: 'Сумма Заказа' },
        { value: 'item_count', label: 'Кол-во Позиций' },
        { value: 'total_quantity', label: 'Общее Кол-во Товаров' },
        { value: 'hour_of_day', label: 'Час Дня' },
        { value: 'day_of_week', label: 'День Недели (0=Пн)' },
    ],
    // --- ДОБАВЛЯЕМ ПРИЗНАКИ ДЛЯ СЕССИЙ ---
    activity_session: [
        { value: 'action_count', label: 'Кол-во Действий' },
        { value: 'session_duration_seconds', label: 'Длительность (сек)' },
        { value: 'unique_action_types', label: 'Уникальных Типов Действий' },
        { value: 'failed_login_count', label: 'Неудачных Логинов' },
    ]
    // -------------------------------------
};
// -------------------------------------------

function AnalysisPage() {
    const [data, setData] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    
    // TODO: Добавить состояние для выбора сущности и признаков
    const [entityType, setEntityType] = useState('order');
    const [featureX, setFeatureX] = useState('total_amount');
    const [featureY, setFeatureY] = useState('item_count');
    const [limit, setLimit] = useState(500);

    const fetchData = useCallback(async () => {
        setLoading(true);
        setError('');
        setData([]);
        try {
            const params = {
                entity_type: entityType,
                feature_x: featureX,
                feature_y: featureY,
                limit: limit,
            };
            console.log('Fetching scatter plot data with params:', params);
            const response = await axios.get('/api/charts/feature_scatter', { params });
            console.log('Received data:', response.data);
            setData(response.data || []);
        } catch (err) {
            console.error("Error fetching scatter plot data:", err);
            setError(err.response?.data?.detail || err.message || 'Не удалось загрузить данные для диаграммы рассеяния.');
        } finally {
            setLoading(false);
        }
    }, [entityType, featureX, featureY, limit]);

    // Загрузка данных при первой загрузке (или изменении параметров, если добавить зависимости)
    useEffect(() => {
        fetchData();
    }, [fetchData]);

    return (
        <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
            <Typography variant="h4" gutterBottom>
                Анализ Признаков (Scatter Plot)
            </Typography>
            
            <Paper sx={{ p: 3, mb: 3 }}>
                <Typography variant="h6" gutterBottom>Настройки графика</Typography>
                <Box sx={{ display: 'flex', gap: 2, mb: 2, flexWrap: 'wrap' }}> {/* Добавим flexWrap */}
                     <FormControl size="small" sx={{ minWidth: 150 }}>
                        <InputLabel>Тип Сущности</InputLabel>
                        <Select value={entityType} label="Тип Сущности" onChange={(e) => {
                            const newType = e.target.value;
                            setEntityType(newType);
                            // Сбрасываем признаки на дефолтные для нового типа
                            if (newType === 'order') {
                                setFeatureX('total_amount');
                                setFeatureY('item_count');
                            } else if (newType === 'activity_session') {
                                setFeatureX('action_count');
                                setFeatureY('session_duration_seconds');
                            }
                        }}> 
                            <MenuItem value="order">Заказы (Order)</MenuItem>
                            <MenuItem value="activity_session">Сессии (Activity)</MenuItem>
                        </Select>
                    </FormControl>
                    <FormControl size="small" sx={{ minWidth: 200 }}>
                        <InputLabel>Признак X</InputLabel>
                         <Select value={featureX} label="Признак X" onChange={(e) => setFeatureX(e.target.value)}> 
                            {featureOptions[entityType]?.map(option => (
                                <MenuItem key={option.value} value={option.value}>
                                    {option.label}
                                </MenuItem>
                            ))}
                        </Select>
                    </FormControl>
                    <FormControl size="small" sx={{ minWidth: 200 }}>
                        <InputLabel>Признак Y</InputLabel>
                         <Select value={featureY} label="Признак Y" onChange={(e) => setFeatureY(e.target.value)}> 
                             {featureOptions[entityType]?.map(option => (
                                <MenuItem key={option.value} value={option.value}>
                                    {option.label}
                                </MenuItem>
                            ))}
                        </Select>
                    </FormControl>
                     <Button variant="contained" onClick={fetchData} disabled={loading}>
                        {loading ? <CircularProgress size={24} /> : 'Обновить Данные'}
                    </Button>
                </Box>
            </Paper>

            <Paper sx={{ p: 2, height: '60vh', position: 'relative' }}> {/* Задаем высоту */}
                {loading && (
                    <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                        <CircularProgress />
                    </Box>
                )}
                {error && (
                     <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                         <Alert severity="error" sx={{ width: '80%' }}>{error}</Alert>
                     </Box>
                )}
                {!loading && !error && data.length === 0 && (
                     <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                        <Typography color="text.secondary">Нет данных для отображения.</Typography>
                     </Box>
                )}
                {!loading && !error && data.length > 0 && (
                    <FeatureScatterPlot data={data} featureXName={featureX} featureYName={featureY} />
                )}
            </Paper>
        </Container>
    );
}

export default AnalysisPage; 
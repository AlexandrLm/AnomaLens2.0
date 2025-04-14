import React, { useState, useEffect, useCallback, useMemo } from 'react';
import axios from 'axios';
import {
    Box,
    Typography,
    Paper,
    CircularProgress,
    Alert,
    useTheme,
    Grid,
    Card,
    CardContent,
    CardActionArea,
    TextField,
    InputAdornment,
    Chip,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    IconButton,
    Button,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableRow,
    TablePagination
} from '@mui/material';
import { Search as SearchIcon, Close as CloseIcon } from '@mui/icons-material';
import { format } from 'date-fns';
import { ru } from 'date-fns/locale';

// --- Компонент Карточки Активности --- 
const ActivityCard = React.memo(({ activity, onClick }) => {
    const theme = useTheme();

    const activityId = activity?.id ?? 'N/A';
    const customerId = activity?.customer_id ?? 'N/A';
    const actionType = activity?.action_type ?? 'N/A';
    const ipAddress = activity?.ip_address ?? 'N/A';

    let timestampFormatted = 'Invalid Date';
    if (activity?.timestamp) {
        try {
            const date = new Date(activity.timestamp);
            if (!isNaN(date.getTime())) {
                timestampFormatted = format(date, 'dd.MM.yy HH:mm:ss', { locale: ru });
            }
        } catch { /* Оставляем 'Invalid Date' */ }
    }

    // Определяем цвет чипа в зависимости от типа действия (пример)
    let chipColor = 'default';
    if (actionType.toLowerCase().includes('login')) chipColor = 'success';
    else if (actionType.toLowerCase().includes('fail')) chipColor = 'error';
    else if (actionType.toLowerCase().includes('view')) chipColor = 'info';
    else if (actionType.toLowerCase().includes('add')) chipColor = 'primary';

    return (
        <Grid item xs={12} sm={6} md={4} lg={3}> 
            <Card 
                elevation={2} 
                sx={{ 
                    height: '100%', 
                    display: 'flex', 
                    flexDirection: 'column',
                    transition: 'transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out',
                    '&:hover': {
                        transform: 'translateY(-4px)',
                        boxShadow: theme.shadows[4]
                    }
                }}
            >
                <CardActionArea onClick={() => onClick(activity)} sx={{ flexGrow: 1 }}>
                    <CardContent>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                            <Typography variant="caption" color="text.secondary">
                                ID: {activityId}
                            </Typography>
                             <Chip label={actionType} size="small" color={chipColor} />
                        </Box>
                        <Typography variant="body2" sx={{ mb: 0.5 }}>
                            Пользователь ID: {customerId}
                        </Typography>
                         <Typography variant="body2" color="text.secondary" sx={{ mb: 0.5 }}>
                            IP: {ipAddress}
                        </Typography>
                        <Typography variant="caption" color="text.secondary" display="block"> {/* Время внизу */} 
                            {timestampFormatted}
                        </Typography>
                    </CardContent>
                </CardActionArea>
            </Card>
        </Grid>
    );
});

// --- Компонент для отображения деталей активности в модальном окне --- 
const ActivityDetailsContent = ({ activity }) => {
    if (!activity) return null;

    let timestampFormatted = 'N/A';
    if (activity?.timestamp) {
        try {
            const date = new Date(activity.timestamp);
            if (!isNaN(date.getTime())) {
                timestampFormatted = date.toLocaleString('ru-RU', { dateStyle: 'full', timeStyle: 'medium' });
            }
        } catch { /* N/A */ }
    }

    // --- Улучшенное отображение Деталей --- 
    let detailsContent = <Typography variant="body2" color="text.secondary">Нет данных</Typography>;
    if (activity?.details !== null && activity?.details !== undefined) {
        if (typeof activity.details === 'object') {
            // Если это объект, выводим как таблицу ключ-значение
            detailsContent = (
                <TableContainer component={Paper} elevation={0} variant="outlined">
                    <Table size="small">
                        <TableBody>
                            {Object.entries(activity.details).map(([key, value]) => (
                                <TableRow key={key}>
                                    <TableCell sx={{fontWeight: 'bold', width: '40%'}}>{String(key)}:</TableCell>
                                    <TableCell>{String(value)}</TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                </TableContainer>
            );
        } else {
            // Если не объект, выводим как текст в блоке pre
            detailsContent = (
                <Paper elevation={0} variant="outlined" sx={{ p: 1.5, maxHeight: 400, overflow: 'auto', bgcolor: 'grey.50' }}>
                     <pre style={{ margin: 0, fontSize: '0.8rem', whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
                         {String(activity.details)}
                     </pre>
                </Paper>
            );
        }
    }
    // ---------------------------------------

    return (
        <Grid container spacing={2}> 
             {/* Колонка 1: Основная информация */} 
            <Grid item xs={12} md={5}>
                <TableContainer component={Paper} elevation={0} variant="outlined">
                    <Table size="small">
                        <TableBody>
                            <TableRow><TableCell sx={{fontWeight: 'bold'}}>ID Активности:</TableCell><TableCell>{activity.id ?? 'N/A'}</TableCell></TableRow>
                            <TableRow><TableCell sx={{fontWeight: 'bold'}}>Время:</TableCell><TableCell>{timestampFormatted}</TableCell></TableRow>
                            <TableRow><TableCell sx={{fontWeight: 'bold'}}>Тип Действия:</TableCell><TableCell>{activity.action_type ?? 'N/A'}</TableCell></TableRow>
                            <TableRow><TableCell sx={{fontWeight: 'bold'}}>ID Пользователя:</TableCell><TableCell>{activity.customer_id ?? 'N/A'}</TableCell></TableRow>
                            <TableRow><TableCell sx={{fontWeight: 'bold'}}>IP Адрес:</TableCell><TableCell>{activity.ip_address ?? 'N/A'}</TableCell></TableRow>
                        </TableBody>
                    </Table>
                </TableContainer>
            </Grid>
             {/* Колонка 2: Детали (теперь рендерится detailsContent) */} 
            <Grid item xs={12} md={7}>
                <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 'medium' }}>Детали</Typography>
                {/* Вставляем отформатированный контент */} 
                {detailsContent}
            </Grid>
        </Grid>
    );
};


// --- Основной компонент страницы --- 
function ActivitiesPage() {
    const theme = useTheme();
    const [activities, setActivities] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');
    const [searchTerm, setSearchTerm] = useState('');
    
    // --- Состояние для пагинации --- 
    const [page, setPage] = useState(0); // Номер страницы (начинается с 0)
    const [rowsPerPage, setRowsPerPage] = useState(24); // Количество активностей на странице (кратно 12, 6, 4, 3)
    const [totalCount, setTotalCount] = useState(0); // Общее количество активностей
    // --------------------------------

    const [isDetailModalOpen, setIsDetailModalOpen] = useState(false);
    const [selectedActivity, setSelectedActivity] = useState(null);

    // --- Функция загрузки данных (с пагинацией) --- 
    const fetchActivities = useCallback(async (currentPage, currentRowsPerPage) => {
        setLoading(true);
        setError('');
        try {
            // Рассчитываем skip
            const skip = currentPage * currentRowsPerPage;
            // Формируем URL с параметрами пагинации
            const apiUrl = `/api/activities/?limit=${currentRowsPerPage}&skip=${skip}`;
            console.log("Fetching activities from:", apiUrl); // Лог URL

            const response = await axios.get(apiUrl);
            
            // Ожидаем объект { totalCount: number, data: array }
            if (response?.data && typeof response.data.totalCount === 'number' && Array.isArray(response.data.data)) {
                console.log(`Received ${response.data.data.length} activities, total count: ${response.data.totalCount}`);
                setActivities(response.data.data);
                setTotalCount(response.data.totalCount);
            } else {
                 console.error('Received unexpected data structure for activities:', response?.data);
                 setError('Получен некорректный формат данных от API.');
                 setActivities([]);
                 setTotalCount(0);
            }
        } catch (err) {
            console.error("Error fetching activities:", err);
            let errorMsg = 'Не удалось загрузить список активностей.';
             if (err.response?.data?.detail) {
                const detail = err.response.data.detail;
                if (typeof detail === 'string') errorMsg = detail;
                else if (Array.isArray(detail) && detail.length > 0 && typeof detail[0].msg === 'string') errorMsg = detail[0].msg;
                else errorMsg = JSON.stringify(detail);
            } else if (err.message) errorMsg = err.message;
            setError(errorMsg);
            setActivities([]);
            setTotalCount(0);
        } finally {
            setLoading(false);
        }
    }, []); // Зависимости будут добавлены через useEffect

    useEffect(() => {
        // Вызываем fetchActivities при монтировании и при изменении страницы или кол-ва строк
        fetchActivities(page, rowsPerPage);
    }, [fetchActivities, page, rowsPerPage]);

    // --- Обработчики пагинации --- 
    const handleChangePage = (event, newPage) => {
      console.log("Changing page to:", newPage);
      setPage(newPage);
      // fetchActivities будет вызван из useEffect
    };

    const handleChangeRowsPerPage = (event) => {
      const newRowsPerPage = parseInt(event.target.value, 10);
      console.log("Changing rows per page to:", newRowsPerPage);
      setRowsPerPage(newRowsPerPage);
      setPage(0); // Сбрасываем на первую страницу при изменении кол-ва строк
      // fetchActivities будет вызван из useEffect
    };
    // ----------------------------

    // --- Фильтрация активностей (теперь фильтрует только на текущей странице) --- 
    const handleSearchChange = (event) => {
        setSearchTerm(event.target.value);
    };

    const filteredActivities = useMemo(() => {
        const lowerCaseSearchTerm = searchTerm.toLowerCase().trim();
        if (!lowerCaseSearchTerm) {
            return activities;
        }
        return activities.filter(activity => {
            const actionType = String(activity?.action_type ?? '').toLowerCase();
            const customerId = String(activity?.customer_id ?? '').toLowerCase();
            const ipAddress = String(activity?.ip_address ?? '').toLowerCase();
            const id = String(activity?.id ?? '').toLowerCase();
            // Ищем по основным полям
            return actionType.includes(lowerCaseSearchTerm) || 
                   customerId.includes(lowerCaseSearchTerm) || 
                   ipAddress.includes(lowerCaseSearchTerm) ||
                   id.includes(lowerCaseSearchTerm);
        });
    }, [activities, searchTerm]);

    // --- Обработчик клика по карточке --- 
    const handleCardClick = useCallback((activity) => {
        setSelectedActivity(activity);
        setIsDetailModalOpen(true);  
    }, []);

    // --- Функция закрытия модального окна --- 
    const handleCloseDetailModal = () => {
        setIsDetailModalOpen(false);
        setSelectedActivity(null);
    };

    return (
        <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%', p: 3, gap: 2 }}>
            <Typography
                variant="h4"
                component="h1"
                sx={{ fontWeight: 600, color: 'text.primary'}}
            >
                Активности Пользователей
            </Typography>

            {/* --- Панель Фильтров --- */}
            <Paper elevation={1} sx={{ p: 2, borderRadius: 2, mb: 1 }}>
                 <TextField
                    fullWidth
                    variant="outlined"
                    size="small"
                    placeholder="Поиск по ID, Типу, ID Пользователя или IP..."
                    value={searchTerm}
                    onChange={handleSearchChange}
                    InputProps={{
                        startAdornment: (
                            <InputAdornment position="start">
                                <SearchIcon color="action" />
                            </InputAdornment>
                        ),
                    }}
                />
            </Paper>

            {/* --- Отображение Карточек или Сообщений --- */}
            <Box sx={{ flexGrow: 1, overflowY: 'auto' }}>
                {loading && (
                    <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '50%' }}>
                        <CircularProgress />
                    </Box>
                )}
                {!loading && error && (
                    <Alert severity="error" sx={{ m: 2, borderRadius: 1 }}>
                        {error}
                    </Alert>
                )}
                {!loading && !error && filteredActivities.length === 0 && (
                    <Typography sx={{ textAlign: 'center', color: 'text.secondary', mt: 4 }}>
                        {searchTerm ? 'Активности не найдены' : 'Нет активностей для отображения'}
                    </Typography>
                )}
                {!loading && !error && filteredActivities.length > 0 && (
                    <Grid container spacing={2}>
                        {filteredActivities.map((activity) => (
                            <ActivityCard key={activity?.id ?? Math.random()} activity={activity} onClick={handleCardClick} />
                        ))}
                    </Grid>
                )}
            </Box>

            {/* Компонент пагинации */}
            {!loading && !error && totalCount > 0 && (
                <TablePagination
                    component="div"
                    count={totalCount} // Общее количество активностей
                    page={page} // Текущая страница (0-based)
                    onPageChange={handleChangePage} // Обработчик смены страницы
                    rowsPerPage={rowsPerPage} // Количество строк на странице
                    onRowsPerPageChange={handleChangeRowsPerPage} // Обработчик смены кол-ва строк
                    rowsPerPageOptions={[12, 24, 48, 96]} // Варианты кол-ва строк
                    labelRowsPerPage="Активностей на странице:"
                    labelDisplayedRows={({ from, to, count }) => `${from}–${to} из ${count}`}
                    sx={{ mt: 3, borderTop: '1px solid', borderColor: 'divider' }}
                />
            )}

            {/* === МОДАЛЬНОЕ ОКНО ДЕТАЛЕЙ АКТИВНОСТИ === */}
            <Dialog
                open={isDetailModalOpen}
                onClose={handleCloseDetailModal}
                maxWidth="md" // Средний размер окна
                fullWidth
                aria-labelledby="activity-detail-dialog-title"
            >
                <DialogTitle id="activity-detail-dialog-title">
                    Детали Активности #{selectedActivity?.id ?? ''}
                    <IconButton
                        aria-label="close"
                        onClick={handleCloseDetailModal}
                        sx={{
                            position: 'absolute',
                            right: 8,
                            top: 8,
                            color: (theme) => theme.palette.grey[500],
                        }}
                    >
                        <CloseIcon />
                    </IconButton>
                </DialogTitle>
                <DialogContent dividers> 
                    {selectedActivity && <ActivityDetailsContent activity={selectedActivity} />}
                </DialogContent>
                <DialogActions>
                    <Button onClick={handleCloseDetailModal}>Закрыть</Button>
                </DialogActions>
            </Dialog>
            {/* ========================================== */}
        </Box>
    );
}

export default ActivitiesPage; 
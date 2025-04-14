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
    TableHead,
    TableRow
} from '@mui/material';
import { Search as SearchIcon, Close as CloseIcon } from '@mui/icons-material';
import { format } from 'date-fns';
import { ru } from 'date-fns/locale';

// --- Компонент Карточки Заказа --- 
const OrderCard = React.memo(({ order, onClick }) => {
    const theme = useTheme();

    // Безопасное получение и форматирование данных
    const orderId = order?.id ?? 'N/A';
    const customerId = order?.customer_id ?? 'N/A';
    const itemCount = Array.isArray(order?.items) ? order.items.length : 0;
    
    let orderDateFormatted = 'Invalid Date';
    if (order?.created_at) {
        try {
            const date = new Date(order.created_at);
            if (!isNaN(date.getTime())) {
                orderDateFormatted = format(date, 'dd MMM yyyy, HH:mm', { locale: ru });
            }
        } catch { /* Оставляем 'Invalid Date' */ }
    }

    let totalAmountFormatted = 'N/A';
    if (typeof order?.total_amount === 'number') {
        try {
            totalAmountFormatted = order.total_amount.toLocaleString('ru-RU', { style: 'currency', currency: 'RUB' });
        } catch { /* Оставляем 'N/A' */ }
    }

    return (
        <Grid item xs={12} sm={6} md={4} lg={3}> {/* Адаптивная сетка */} 
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
                <CardActionArea onClick={() => onClick(order)} sx={{ flexGrow: 1 }}>
                    <CardContent sx={{ pb: 1 }}>
                        <Typography variant="h6" component="div" gutterBottom noWrap>
                            Заказ #{orderId}
                        </Typography>
                        <Typography sx={{ mb: 0.5 }} color="text.secondary" variant="body2">
                            Покупатель ID: {customerId}
                        </Typography>
                        <Typography sx={{ mb: 0.5 }} color="text.secondary" variant="body2">
                            Дата: {orderDateFormatted}
                        </Typography>
                        <Typography sx={{ mb: 0.5 }} variant="body1" fontWeight="medium">
                            Сумма: {totalAmountFormatted}
                        </Typography>
                        <Typography color="text.secondary" variant="body2">
                            Позиций: {itemCount}
                        </Typography>
                    </CardContent>
                </CardActionArea>
            </Card>
        </Grid>
    );
});

// --- Компонент для отображения деталей заказа ВНУТРИ МОДАЛЬНОГО ОКНА --- 
// (Можно переиспользовать и доработать логику из OrderDetailPanel, который был раньше)
const OrderDetailsContent = ({ order }) => {
    if (!order) return null;

    // Безопасный подсчет суммы по позициям
    const calculatedTotal = Array.isArray(order.items) ? order.items.reduce((sum, item) => {
        const price = typeof item.price === 'number' ? item.price : 0;
        const quantity = typeof item.quantity === 'number' ? item.quantity : 0;
        return sum + (price * quantity);
    }, 0) : 0;

    const orderTotalAmount = typeof order.total_amount === 'number' ? order.total_amount : 0;
    const totalsMismatch = Math.abs(calculatedTotal - orderTotalAmount) > 0.01;

    // Форматируем дату для деталей
    let orderDateFormatted = 'Invalid Date';
    if (order?.created_at) {
        try {
            const date = new Date(order.created_at);
            if (!isNaN(date.getTime())) {
                orderDateFormatted = format(date, 'dd MMMM yyyy HH:mm:ss', { locale: ru });
            }
        } catch { /* Оставляем 'Invalid Date' */ }
    }

    return (
        <Grid container spacing={2}>
            {/* Колонка с основной информацией о заказе */}
            <Grid item xs={12} md={5}>
                <Typography variant="h6" gutterBottom>Детали Заказа</Typography>
                <TableContainer component={Paper} elevation={0} variant="outlined">
                    <Table size="small">
                        <TableBody>
                            <TableRow><TableCell sx={{fontWeight: 'bold'}}>ID Заказа:</TableCell><TableCell>{order.id}</TableCell></TableRow>
                            <TableRow><TableCell sx={{fontWeight: 'bold'}}>ID Покупателя:</TableCell><TableCell>{order.customer_id}</TableCell></TableRow>
                            <TableRow><TableCell sx={{fontWeight: 'bold'}}>Дата Создания:</TableCell><TableCell>{orderDateFormatted}</TableCell></TableRow>
                            <TableRow><TableCell sx={{fontWeight: 'bold'}}>Сумма Заказа:</TableCell><TableCell sx={{fontWeight: 'bold'}}>{orderTotalAmount.toLocaleString('ru-RU', { style: 'currency', currency: 'RUB' })}</TableCell></TableRow>
                        </TableBody>
                    </Table>
                </TableContainer>
            </Grid>

            {/* Колонка со списком товаров */}
            <Grid item xs={12} md={7}>
                 <Typography variant="h6" gutterBottom>Состав Заказа</Typography>
                 {Array.isArray(order.items) && order.items.length > 0 ? (
                    <TableContainer component={Paper} elevation={0} variant="outlined">
                        <Table size="small" aria-label="order items">
                            <TableHead>
                                <TableRow sx={{ '& th': { fontWeight: 'bold', bgcolor: 'action.hover' } }}>
                                    <TableCell>ID Продукта</TableCell>
                                    <TableCell align="right">Кол-во</TableCell>
                                    <TableCell align="right">Цена за шт.</TableCell>
                                    <TableCell align="right">Сумма</TableCell>
                                </TableRow>
                            </TableHead>
                            <TableBody>
                                {order.items.map((item, index) => {
                                    const price = typeof item.price === 'number' ? item.price : 0;
                                    const quantity = typeof item.quantity === 'number' ? item.quantity : 0;
                                    const itemSum = price * quantity;
                                    return (
                                        <TableRow key={`${order.id}-${item?.product_id ?? index}-${index}`}>
                                            <TableCell component="th" scope="row">{item?.product_id ?? 'N/A'}</TableCell>
                                            <TableCell align="right">{quantity}</TableCell>
                                            <TableCell align="right">{price.toLocaleString('ru-RU', { style: 'currency', currency: 'RUB' })}</TableCell>
                                            <TableCell align="right">{itemSum.toLocaleString('ru-RU', { style: 'currency', currency: 'RUB' })}</TableCell>
                                        </TableRow>
                                    );
                                })}
                                <TableRow sx={{ '& td': { fontWeight: 'bold', borderTop: '2px solid', borderColor: 'divider' } }}>
                                    <TableCell colSpan={3} align="right">Итого по позициям:</TableCell>
                                    <TableCell align="right">{calculatedTotal.toLocaleString('ru-RU', { style: 'currency', currency: 'RUB' })}</TableCell>
                                </TableRow>
                            </TableBody>
                        </Table>
                    </TableContainer>
                 ) : (
                    <Typography color="text.secondary" sx={{mt: 2}}>Нет информации о позициях.</Typography>
                 )}
                 {totalsMismatch && (
                    <Alert severity="warning" sx={{ mt: 1.5, fontSize: '0.8rem' }}>
                        Внимание: Сумма позиций не совпадает с общей суммой заказа.
                    </Alert>
                )}
            </Grid>
        </Grid>
    );
};

// --- Основной компонент страницы --- 
function OrdersPage() {
    const theme = useTheme();
    const [orders, setOrders] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');
    const [searchTerm, setSearchTerm] = useState('');
    
    // --- Состояние для модального окна --- 
    const [isDetailModalOpen, setIsDetailModalOpen] = useState(false);
    const [selectedOrder, setSelectedOrder] = useState(null);
    // -------------------------------------

    // --- Функция загрузки данных (без изменений) --- 
    const fetchOrders = useCallback(async () => {
        setLoading(true);
        setError('');
        try {
            const response = await axios.get('/api/store/orders/?limit=1000'); 
            if (Array.isArray(response?.data)) {
                if (response.data.length > 0) {
                    console.log("Структура первого заказа из API:", response.data[0]);
                }
                setOrders(response.data);
            } else {
                console.error('Received non-array data for orders:', response?.data);
                setError('Получен некорректный формат данных.');
                setOrders([]);
            }
        } catch (err) {
            console.error("Error fetching orders:", err);
            let errorMsg = 'Не удалось загрузить список заказов.';
            if (err.response?.data?.detail) {
                const detail = err.response.data.detail;
                if (typeof detail === 'string') errorMsg = detail;
                else if (Array.isArray(detail) && detail.length > 0 && typeof detail[0].msg === 'string') errorMsg = detail[0].msg;
                else errorMsg = JSON.stringify(detail);
            } else if (err.message) errorMsg = err.message;
            setError(errorMsg);
            setOrders([]);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchOrders();
    }, [fetchOrders]);

    // --- Фильтрация заказов --- 
    const handleSearchChange = (event) => {
        setSearchTerm(event.target.value);
    };

    const filteredOrders = useMemo(() => {
        const lowerCaseSearchTerm = searchTerm.toLowerCase().trim();
        if (!lowerCaseSearchTerm) {
            return orders; // Если поиск пуст, показываем все
        }
        return orders.filter(order => {
            const orderId = String(order?.id ?? '').toLowerCase();
            const customerId = String(order?.customer_id ?? '').toLowerCase();
            return orderId.includes(lowerCaseSearchTerm) || customerId.includes(lowerCaseSearchTerm);
        });
    }, [orders, searchTerm]);

    // --- ОБНОВЛЕННЫЙ Обработчик клика по карточке --- 
    const handleCardClick = useCallback((order) => {
        setSelectedOrder(order); // Сохраняем данные выбранного заказа
        setIsDetailModalOpen(true); // Открываем модальное окно
        // console.log("Order clicked:", order); 
        // alert(`Вы кликнули на заказ #${order.id}`);
    }, []);
    // ----------------------------------------------

    // --- Функция закрытия модального окна --- 
    const handleCloseDetailModal = () => {
        setIsDetailModalOpen(false);
        setSelectedOrder(null); // Сбрасываем выбранный заказ
    };
    // ---------------------------------------

    return (
        <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%', p: 3, gap: 2 }}>
            <Typography
                variant="h4"
                component="h1"
                sx={{ fontWeight: 600, color: 'text.primary'}}
            >
                Заказы
            </Typography>

            {/* --- Панель Фильтров --- */}
            <Paper elevation={1} sx={{ p: 2, borderRadius: 2, mb: 1 }}>
                 <TextField
                    fullWidth
                    variant="outlined"
                    size="small"
                    placeholder="Поиск по ID Заказа или ID Покупателя..."
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
                {/* Сюда можно добавить другие фильтры (по дате, статусу и т.д.) */} 
            </Paper>

            {/* --- Отображение Карточек или Сообщений --- */}
            <Box sx={{ flexGrow: 1, overflowY: 'auto' /* Добавляем скролл */ }}>
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
                {!loading && !error && filteredOrders.length === 0 && (
                    <Typography sx={{ textAlign: 'center', color: 'text.secondary', mt: 4 }}>
                        {searchTerm ? 'Заказы не найдены' : 'Нет заказов для отображения'}
                    </Typography>
                )}
                {!loading && !error && filteredOrders.length > 0 && (
                    <Grid container spacing={2}> {/* Используем spacing={2} для отступов */} 
                        {filteredOrders.map((order) => (
                             // Убедимся, что key уникален и стабилен
                            <OrderCard key={order?.id ?? Math.random()} order={order} onClick={handleCardClick} />
                        ))}
                    </Grid>
                )}
            </Box>

            {/* === МОДАЛЬНОЕ ОКНО ДЕТАЛЕЙ ЗАКАЗА === */}
            <Dialog
                open={isDetailModalOpen}
                onClose={handleCloseDetailModal}
                maxWidth="lg" // Делаем окно шире для таблицы товаров
                fullWidth
                aria-labelledby="order-detail-dialog-title"
            >
                <DialogTitle id="order-detail-dialog-title">
                    Детали Заказа #{selectedOrder?.id ?? ''}
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
                <DialogContent dividers> {/* Добавляем разделители */} 
                    {/* Отображаем контент только если есть выбранный заказ */}
                    {selectedOrder && <OrderDetailsContent order={selectedOrder} />}
                </DialogContent>
                <DialogActions>
                    <Button onClick={handleCloseDetailModal}>Закрыть</Button>
                </DialogActions>
            </Dialog>
            {/* ======================================= */}
        </Box>
    );
}

export default OrdersPage;
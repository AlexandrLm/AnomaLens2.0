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
    TableRow
} from '@mui/material';
import { Search as SearchIcon, Close as CloseIcon } from '@mui/icons-material';

// --- Компонент Карточки Пользователя --- 
const CustomerCard = React.memo(({ customer, onClick }) => {
    const theme = useTheme();

    const customerId = customer?.id ?? 'N/A';
    const firstName = customer?.first_name ?? '';
    const lastName = customer?.last_name ?? '';
    const customerFullName = `${firstName} ${lastName}`.trim() || 'Имя не указано';
    const customerEmail = customer?.email ?? 'Email не указан';

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
                <CardActionArea onClick={() => onClick(customer)} sx={{ flexGrow: 1 }}>
                    <CardContent>
                        <Typography variant="h6" component="div" noWrap sx={{ mb: 0.5 }} title={customerFullName}>
                            {customerFullName}
                        </Typography>
                        <Typography sx={{ mb: 1 }} color="text.secondary" variant="body2">
                            ID: {customerId}
                        </Typography>
                        <Typography color="text.secondary" variant="body2" noWrap title={customerEmail}>
                            {customerEmail}
                        </Typography>
                    </CardContent>
                </CardActionArea>
            </Card>
        </Grid>
    );
});

// --- Компонент для отображения деталей пользователя в модальном окне --- 
const CustomerDetailsContent = ({ customer }) => {
    if (!customer) return null;

    const firstName = customer?.first_name ?? 'N/A';
    const lastName = customer?.last_name ?? 'N/A';

    let createdAtFormatted = 'N/A';
    if (customer?.created_at) {
        try {
            const date = new Date(customer.created_at);
            if (!isNaN(date.getTime())) {
                createdAtFormatted = date.toLocaleString('ru-RU'); 
            }
        } catch { /* Оставляем N/A */ }
    }

    return (
         <TableContainer component={Paper} elevation={0} variant="outlined">
            <Table size="small">
                <TableBody>
                    <TableRow><TableCell sx={{fontWeight: 'bold'}}>ID:</TableCell><TableCell>{customer.id ?? 'N/A'}</TableCell></TableRow>
                    <TableRow><TableCell sx={{fontWeight: 'bold'}}>Имя:</TableCell><TableCell>{firstName}</TableCell></TableRow>
                    <TableRow><TableCell sx={{fontWeight: 'bold'}}>Фамилия:</TableCell><TableCell>{lastName}</TableCell></TableRow>
                    <TableRow><TableCell sx={{fontWeight: 'bold'}}>Email:</TableCell><TableCell>{customer.email ?? 'N/A'}</TableCell></TableRow>
                    <TableRow><TableCell sx={{fontWeight: 'bold'}}>Создан:</TableCell><TableCell>{createdAtFormatted}</TableCell></TableRow>
                </TableBody>
            </Table>
        </TableContainer>
    );
};

// --- Основной компонент страницы --- 
function CustomersPage() {
    const theme = useTheme();
    const [customers, setCustomers] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');
    const [searchTerm, setSearchTerm] = useState('');
    
    // --- Состояние для модального окна --- 
    const [isDetailModalOpen, setIsDetailModalOpen] = useState(false);
    const [selectedCustomer, setSelectedCustomer] = useState(null);
    // -------------------------------------

    // --- Функция загрузки данных (немного улучшим обработку ошибок) --- 
    const fetchCustomers = useCallback(async () => {
        setLoading(true);
        setError('');
        try {
            const response = await axios.get('/api/store/customers/?limit=1000');
            if (Array.isArray(response?.data)) {
                 if (response.data.length > 0) {
                     console.log("Структура первого пользователя из API:", response.data[0]);
                 }
                setCustomers(response.data);
            } else {
                 console.error('Received non-array data for customers:', response?.data);
                 setError('Получен некорректный формат данных.');
                 setCustomers([]);
            }
        } catch (err) {
            console.error("Error fetching customers:", err);
            let errorMsg = 'Не удалось загрузить список пользователей.';
            if (err.response?.data?.detail) {
                const detail = err.response.data.detail;
                if (typeof detail === 'string') errorMsg = detail;
                else if (Array.isArray(detail) && detail.length > 0 && typeof detail[0].msg === 'string') errorMsg = detail[0].msg;
                else errorMsg = JSON.stringify(detail);
            } else if (err.message) errorMsg = err.message;
            setError(errorMsg);
            setCustomers([]);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchCustomers();
    }, [fetchCustomers]);

    // --- Фильтрация пользователей --- 
    const handleSearchChange = (event) => {
        setSearchTerm(event.target.value);
    };

    const filteredCustomers = useMemo(() => {
        const lowerCaseSearchTerm = searchTerm.toLowerCase().trim();
        if (!lowerCaseSearchTerm) {
            return customers;
        }
        return customers.filter(customer => {
            const firstName = String(customer?.first_name ?? '').toLowerCase();
            const lastName = String(customer?.last_name ?? '').toLowerCase();
            const email = String(customer?.email ?? '').toLowerCase();
            const id = String(customer?.id ?? '').toLowerCase();
            return firstName.includes(lowerCaseSearchTerm) || 
                   lastName.includes(lowerCaseSearchTerm) || 
                   `${firstName} ${lastName}`.includes(lowerCaseSearchTerm) ||
                   email.includes(lowerCaseSearchTerm) || 
                   id.includes(lowerCaseSearchTerm);
        });
    }, [customers, searchTerm]);

    // --- ОБНОВЛЕННЫЙ Обработчик клика по карточке --- 
    const handleCardClick = useCallback((customer) => {
        setSelectedCustomer(customer);
        setIsDetailModalOpen(true);
    }, []);

    // --- Функция закрытия модального окна --- 
    const handleCloseDetailModal = () => {
        setIsDetailModalOpen(false);
        setSelectedCustomer(null);
    };

    return (
        <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%', p: 3, gap: 2 }}>
            <Typography
                variant="h4"
                component="h1"
                sx={{ fontWeight: 600, color: 'text.primary'}}
            >
                Пользователи
            </Typography>

            {/* --- Панель Фильтров --- */}
            <Paper elevation={1} sx={{ p: 2, borderRadius: 2, mb: 1 }}>
                 <TextField
                    fullWidth
                    variant="outlined"
                    size="small"
                    placeholder="Поиск по ID, Имени, Фамилии или Email..."
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
                {!loading && !error && filteredCustomers.length === 0 && (
                    <Typography sx={{ textAlign: 'center', color: 'text.secondary', mt: 4 }}>
                        {searchTerm ? 'Пользователи не найдены' : 'Нет пользователей для отображения'}
                    </Typography>
                )}
                {!loading && !error && filteredCustomers.length > 0 && (
                    <Grid container spacing={2}>
                        {filteredCustomers.map((customer) => (
                            <CustomerCard key={customer?.id ?? Math.random()} customer={customer} onClick={handleCardClick} />
                        ))}
                    </Grid>
                )}
            </Box>

            {/* === МОДАЛЬНОЕ ОКНО ДЕТАЛЕЙ ПОЛЬЗОВАТЕЛЯ === */}
            <Dialog
                open={isDetailModalOpen}
                onClose={handleCloseDetailModal}
                maxWidth="sm"
                fullWidth
                aria-labelledby="customer-detail-dialog-title"
            >
                <DialogTitle id="customer-detail-dialog-title">
                    Детали Пользователя: {`${selectedCustomer?.first_name ?? ''} ${selectedCustomer?.last_name ?? ''}`.trim()}
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
                    {selectedCustomer && <CustomerDetailsContent customer={selectedCustomer} />}
                </DialogContent>
                <DialogActions>
                    <Button onClick={handleCloseDetailModal}>Закрыть</Button>
                </DialogActions>
            </Dialog>
            {/* ========================================== */}

        </Box>
    );
}

export default CustomersPage; 
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
    TableRow,
    alpha // Может понадобиться для стилей
} from '@mui/material';
import { Search as SearchIcon, Close as CloseIcon } from '@mui/icons-material';

// --- Компонент Карточки Продукта --- 
const ProductCard = React.memo(({ product, onClick }) => {
    const theme = useTheme();

    const productId = product?.id ?? 'N/A';
    const productName = product?.name ?? 'Без названия';
    // Отображаем начало описания
    const descriptionStart = (product?.description ?? '').substring(0, 80) + ( (product?.description?.length ?? 0) > 80 ? '...' : ''); 

    let priceFormatted = 'N/A';
    if (typeof product?.price === 'number') {
        try {
            priceFormatted = product.price.toLocaleString('ru-RU', { style: 'currency', currency: 'RUB' });
        } catch { /* N/A */ }
    }

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
                <CardActionArea onClick={() => onClick(product)} sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
                    <CardContent sx={{ flexGrow: 1, width: '100%' }}>
                        <Typography variant="h6" component="div" gutterBottom title={productName}> {/* Title для полного названия */} 
                            {productName}
                        </Typography>
                        <Typography sx={{ mb: 1.5 }} color="text.secondary" variant="body2" title={product?.description ?? ''}>
                            {descriptionStart || 'Нет описания'}
                        </Typography>
                    </CardContent>
                    <Box sx={{ p: 2, pt: 0, width: '100%', textAlign: 'right' }}>
                        <Typography variant="h6" color="primary.main" fontWeight="bold">
                            {priceFormatted}
                        </Typography>
                         <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: -0.5 }}> 
                             ID: {productId}
                         </Typography>
                    </Box>
                </CardActionArea>
            </Card>
        </Grid>
    );
});

// --- Компонент для отображения деталей продукта в модальном окне --- 
const ProductDetailsContent = ({ product }) => {
    if (!product) return null;

     let priceFormatted = 'N/A';
    if (typeof product?.price === 'number') {
        try {
            priceFormatted = product.price.toLocaleString('ru-RU', { style: 'currency', currency: 'RUB' });
        } catch { /* N/A */ }
    }

    // TODO: Добавить другие поля продукта, если они есть (stock, category и т.д.)

    return (
        <Grid container spacing={2}>
             {/* Колонка 1: Основная информация */} 
             <Grid item xs={12}>
                <Typography variant="h6" gutterBottom>Детали Продукта</Typography>
                <TableContainer component={Paper} elevation={0} variant="outlined">
                    <Table size="small">
                        <TableBody>
                            <TableRow><TableCell sx={{fontWeight: 'bold', width: '30%'}}>ID:</TableCell><TableCell>{product.id ?? 'N/A'}</TableCell></TableRow>
                            <TableRow><TableCell sx={{fontWeight: 'bold'}}>Название:</TableCell><TableCell>{product.name ?? 'N/A'}</TableCell></TableRow>
                            <TableRow><TableCell sx={{fontWeight: 'bold'}}>Цена:</TableCell><TableCell sx={{fontWeight: 'bold'}}>{priceFormatted}</TableCell></TableRow>
                            {/* <TableRow><TableCell sx={{fontWeight: 'bold'}}>Категория:</TableCell><TableCell>{product.category ?? 'N/A'}</TableCell></TableRow> */} 
                            {/* <TableRow><TableCell sx={{fontWeight: 'bold'}}>В наличии:</TableCell><TableCell>{product.stock ?? 'N/A'}</TableCell></TableRow> */} 
                        </TableBody>
                    </Table>
                </TableContainer>
            </Grid>
             {/* Колонка 2: Описание */} 
             {product.description && (
                <Grid item xs={12}>
                    <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 'medium', mt: 1 }}>Описание</Typography>
                    <Paper elevation={0} variant="outlined" sx={{ p: 1.5, maxHeight: 250, overflow: 'auto', bgcolor: 'grey.50' }}>
                        <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                            {product.description}
                        </Typography>
                    </Paper>
                </Grid>
             )}
        </Grid>
    );
};


// --- Основной компонент страницы --- 
function ProductsPage() {
    const theme = useTheme();
    const [products, setProducts] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');
    const [searchTerm, setSearchTerm] = useState('');
    
    const [isDetailModalOpen, setIsDetailModalOpen] = useState(false);
    const [selectedProduct, setSelectedProduct] = useState(null);

    // --- Функция загрузки данных (Улучшенная обработка ошибок) --- 
    const fetchProducts = useCallback(async () => {
        setLoading(true);
        setError('');
        try {
            const response = await axios.get('/api/store/products/?limit=1000');
             if (Array.isArray(response?.data)) {
                 if (response.data.length > 0) {
                     console.log("Структура первого продукта из API:", response.data[0]);
                 }
                setProducts(response.data);
            } else {
                 console.error('Received non-array data for products:', response?.data);
                 setError('Получен некорректный формат данных.');
                 setProducts([]);
            }
        } catch (err) {
            console.error("Error fetching products:", err);
            let errorMsg = 'Не удалось загрузить список продуктов.';
             if (err.response?.data?.detail) {
                const detail = err.response.data.detail;
                if (typeof detail === 'string') errorMsg = detail;
                else if (Array.isArray(detail) && detail.length > 0 && typeof detail[0].msg === 'string') errorMsg = detail[0].msg;
                else errorMsg = JSON.stringify(detail);
            } else if (err.message) errorMsg = err.message;
            setError(errorMsg);
            setProducts([]);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchProducts();
    }, [fetchProducts]);

    // --- Фильтрация продуктов --- 
    const handleSearchChange = (event) => {
        setSearchTerm(event.target.value);
    };

    const filteredProducts = useMemo(() => {
        const lowerCaseSearchTerm = searchTerm.toLowerCase().trim();
        if (!lowerCaseSearchTerm) {
            return products;
        }
        return products.filter(product => {
            const name = String(product?.name ?? '').toLowerCase();
            const description = String(product?.description ?? '').toLowerCase();
            const id = String(product?.id ?? '').toLowerCase();
            // Ищем по ID, названию или описанию
            return name.includes(lowerCaseSearchTerm) || 
                   description.includes(lowerCaseSearchTerm) ||
                   id.includes(lowerCaseSearchTerm);
        });
    }, [products, searchTerm]);

    // --- Обработчик клика по карточке --- 
    const handleCardClick = useCallback((product) => {
        setSelectedProduct(product);
        setIsDetailModalOpen(true);  
    }, []);

    // --- Функция закрытия модального окна --- 
    const handleCloseDetailModal = () => {
        setIsDetailModalOpen(false);
        setSelectedProduct(null);
    };

    return (
        <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%', p: 3, gap: 2 }}>
            <Typography
                variant="h4"
                component="h1"
                sx={{ fontWeight: 600, color: 'text.primary'}}
            >
                Каталог Продуктов
            </Typography>

            {/* --- Панель Фильтров --- */}
            <Paper elevation={1} sx={{ p: 2, borderRadius: 2, mb: 1 }}>
                 <TextField
                    fullWidth
                    variant="outlined"
                    size="small"
                    placeholder="Поиск по ID, Названию или Описанию..."
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
                {!loading && !error && filteredProducts.length === 0 && (
                    <Typography sx={{ textAlign: 'center', color: 'text.secondary', mt: 4 }}>
                        {searchTerm ? 'Продукты не найдены' : 'Нет продуктов для отображения'}
                    </Typography>
                )}
                {!loading && !error && filteredProducts.length > 0 && (
                    <Grid container spacing={2}>
                        {filteredProducts.map((product) => (
                            <ProductCard key={product?.id ?? Math.random()} product={product} onClick={handleCardClick} />
                        ))}
                    </Grid>
                )}
            </Box>

            {/* === МОДАЛЬНОЕ ОКНО ДЕТАЛЕЙ ПРОДУКТА === */}
            <Dialog
                open={isDetailModalOpen}
                onClose={handleCloseDetailModal}
                maxWidth="md" // Средний размер
                fullWidth
                aria-labelledby="product-detail-dialog-title"
            >
                <DialogTitle id="product-detail-dialog-title">
                    Детали Продукта: {selectedProduct?.name ?? ''}
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
                    {selectedProduct && <ProductDetailsContent product={selectedProduct} />}
                </DialogContent>
                <DialogActions>
                    <Button onClick={handleCloseDetailModal}>Закрыть</Button>
                </DialogActions>
            </Dialog>
            {/* ========================================== */}
        </Box>
    );
}

export default ProductsPage;
import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import { Box, Typography, Paper, CircularProgress, Alert } from '@mui/material';
import { DataGrid } from '@mui/x-data-grid';

// Колонки для Продуктов
const columns = [
    { field: 'id', headerName: 'ID', width: 90 },
    { field: 'name', headerName: 'Название', width: 250, flex: 1 },
    { field: 'description', headerName: 'Описание', width: 350, flex: 1 },
    {
        field: 'price', 
        headerName: 'Цена', 
        width: 120, 
        type: 'number',
        valueFormatter: (params) => { // Форматируем цену
             if (params.value == null) { return ''; }
             return `${params.value.toFixed(2)} ₽`;
        },
    },
];

function ProductsPage() {
    const [products, setProducts] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    const fetchProducts = useCallback(async () => {
        setLoading(true);
        setError('');
        try {
            // Используем эндпоинт /api/store/products/
            const response = await axios.get('/api/store/products/?limit=1000');
            setProducts(response.data || []);
        } catch (err) {
            console.error("Error fetching products:", err);
            setError('Не удалось загрузить список продуктов.');
            setProducts([]);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchProducts();
    }, [fetchProducts]);

    return (
        <Box sx={{ padding: 3 }}>
            <Typography variant="h4" gutterBottom>Продукты</Typography>
            <Paper elevation={3} sx={{ height: 600, width: '100%' }}>
                {error && <Alert severity="error" sx={{ m: 2 }}>{error}</Alert>}
                <DataGrid
                    rows={products}
                    columns={columns}
                    loading={loading}
                    initialState={{
                        pagination: {
                            paginationModel: { page: 0, pageSize: 100 },
                        },
                    }}
                    pageSizeOptions={[25, 50, 100]}
                    disableRowSelectionOnClick
                    autoHeight={false}
                    sx={{
                         border: 0,
                         '& .MuiDataGrid-columnHeaders': { backgroundColor: 'action.hover' }
                    }}
                />
            </Paper>
        </Box>
    );
}

export default ProductsPage; 
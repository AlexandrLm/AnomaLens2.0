import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import { Box, Typography, Paper, CircularProgress, Alert } from '@mui/material';
import { DataGrid } from '@mui/x-data-grid';

// Колонки для Заказов
const columns = [
    { field: 'id', headerName: 'ID Заказа', width: 120 },
    { field: 'customer_id', headerName: 'ID Покупателя', width: 150 },
    {
        field: 'order_date', 
        headerName: 'Дата Заказа', 
        width: 200, 
        type: 'dateTime',
        valueGetter: (params) => new Date(params.value), // Преобразуем строку в Date для DataGrid
    },
    {
        field: 'total_amount',
        headerName: 'Сумма',
        width: 150,
        type: 'number',
        valueFormatter: (params) => {
             if (params.value == null) { return ''; }
             return `${params.value.toFixed(2)} ₽`;
        },
    },
    {
        field: 'item_count', // Добавим колонку с количеством позиций
        headerName: 'Позиций', 
        width: 120, 
        type: 'number',
        valueGetter: (params) => params.row.items?.length || 0, // Считаем кол-во элементов в items
    },
];

function OrdersPage() {
    const [orders, setOrders] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    const fetchOrders = useCallback(async () => {
        setLoading(true);
        setError('');
        try {
            // Используем эндпоинт /api/store/orders/
            const response = await axios.get('/api/store/orders/?limit=1000');
            setOrders(response.data || []);
        } catch (err) {
            console.error("Error fetching orders:", err);
            setError('Не удалось загрузить список заказов.');
            setOrders([]);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchOrders();
    }, [fetchOrders]);

    return (
        <Box sx={{ padding: 3 }}>
            <Typography variant="h4" gutterBottom>Заказы</Typography>
            <Paper elevation={3} sx={{ height: 600, width: '100%' }}>
                {error && <Alert severity="error" sx={{ m: 2 }}>{error}</Alert>}
                <DataGrid
                    rows={orders}
                    columns={columns}
                    loading={loading}
                    initialState={{
                        pagination: {
                            paginationModel: { page: 0, pageSize: 100 },
                        },
                        // Сортировка по дате заказа по умолчанию (сначала новые)
                        sorting: {
                            sortModel: [{ field: 'order_date', sort: 'desc' }],
                        },
                    }}
                    pageSizeOptions={[25, 50, 100]}
                    disableRowSelectionOnClick
                    autoHeight={false}
                    sx={{
                         border: 0,
                         '& .MuiDataGrid-columnHeaders': { backgroundColor: 'action.hover' }
                    }}
                    // TODO: Добавить возможность разворачивать строку для просмотра items заказа?
                />
            </Paper>
        </Box>
    );
}

export default OrdersPage; 
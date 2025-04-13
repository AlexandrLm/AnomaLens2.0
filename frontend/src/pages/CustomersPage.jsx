import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import { Box, Typography, Paper, CircularProgress, Alert } from '@mui/material';
import { DataGrid } from '@mui/x-data-grid';

// Определяем колонки для таблицы пользователей
const columns = [
    { field: 'id', headerName: 'ID', width: 90 },
    { field: 'name', headerName: 'Имя', width: 250, flex: 1 }, // flex: 1 позволяет колонке растягиваться
    { field: 'email', headerName: 'Email', width: 300, flex: 1 },
    // Можно добавить другие поля, если они есть в схеме Customer
];

function CustomersPage() {
    const [customers, setCustomers] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    // Пагинация (пока не реализуем серверную пагинацию, 
    // DataGrid сделает клиентскую, если данных не очень много)
    // const [page, setPage] = useState(0);
    // const [pageSize, setPageSize] = useState(100);
    // const [rowCount, setRowCount] = useState(0);

    const fetchCustomers = useCallback(async () => {
        setLoading(true);
        setError('');
        try {
            // Загружаем первую "страницу" (например, 100 записей)
            // TODO: Реализовать серверную пагинацию, если записей будет много
            const response = await axios.get('/api/store/customers/?limit=1000'); // Пока грузим до 1000
            setCustomers(response.data || []);
            // setRowCount(response.data.totalCount); // Если API будет возвращать общее кол-во
        } catch (err) {
            console.error("Error fetching customers:", err);
            setError('Не удалось загрузить список пользователей.');
            setCustomers([]);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchCustomers();
    }, [fetchCustomers]);

    return (
        <Box sx={{ padding: 3 }}>
            <Typography variant="h4" gutterBottom>Пользователи</Typography>

            <Paper elevation={3} sx={{ height: 600, width: '100%' }}>
                {error && <Alert severity="error" sx={{ m: 2 }}>{error}</Alert>}
                <DataGrid
                    rows={customers}
                    columns={columns}
                    loading={loading}
                    // Настройки пагинации (клиентская по умолчанию)
                    initialState={{
                        pagination: {
                            paginationModel: { page: 0, pageSize: 100 },
                        },
                    }}
                    pageSizeOptions={[25, 50, 100]}
                    // Функции для серверной пагинации (пока не используются)
                    // paginationMode="server"
                    // rowCount={rowCount}
                    // onPaginationModelChange={(model) => { setPage(model.page); setPageSize(model.pageSize); }}
                    // -------------
                    // Другие опции DataGrid
                    disableRowSelectionOnClick // Отключаем выделение строк по клику
                    autoHeight={false} // Задаем фиксированную высоту через sx у Paper
                    sx={{
                         border: 0, // Убираем рамку самого DataGrid
                         '& .MuiDataGrid-columnHeaders': {
                             backgroundColor: 'action.hover' // Цвет заголовков
                         }
                    }}
                />
            </Paper>
        </Box>
    );
}

export default CustomersPage; 
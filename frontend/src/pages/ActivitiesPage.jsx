import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import { Box, Typography, Paper, CircularProgress, Alert } from '@mui/material';
import { DataGrid } from '@mui/x-data-grid';

// Колонки для Активностей
const columns = [
    { field: 'id', headerName: 'ID', width: 90 },
    {
        field: 'timestamp',
        headerName: 'Время',
        width: 200,
        type: 'dateTime',
        valueGetter: (params) => new Date(params.value),
    },
    { field: 'customer_id', headerName: 'ID Покупателя', width: 130 },
    { field: 'action_type', headerName: 'Тип Действия', width: 180 },
    { field: 'ip_address', headerName: 'IP Адрес', width: 150 },
    {
        field: 'details',
        headerName: 'Детали (JSON)',
        width: 300,
        flex: 1,
        // Отображаем JSON как строку
        valueFormatter: (params) => {
            if (params.value == null) { return ''; }
            try {
                return JSON.stringify(params.value);
            } catch { return String(params.value); }
        },
        // Отключаем сортировку/фильтрацию для JSON поля
        sortable: false,
        filterable: false,
    },
];

function ActivitiesPage() {
    const [activities, setActivities] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    const fetchActivities = useCallback(async () => {
        setLoading(true);
        setError('');
        try {
            // Используем эндпоинт /api/activities/
            const response = await axios.get('/api/activities/?limit=1000');
            setActivities(response.data || []);
        } catch (err) {
            console.error("Error fetching activities:", err);
            setError('Не удалось загрузить список активностей.');
            setActivities([]);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchActivities();
    }, [fetchActivities]);

    return (
        <Box sx={{ padding: 3 }}>
            <Typography variant="h4" gutterBottom>Активности Пользователей</Typography>
            <Paper elevation={3} sx={{ height: 700, width: '100%' }}> {/* Увеличим высоту */}
                {error && <Alert severity="error" sx={{ m: 2 }}>{error}</Alert>}
                <DataGrid
                    rows={activities}
                    columns={columns}
                    loading={loading}
                    initialState={{
                        pagination: {
                            paginationModel: { page: 0, pageSize: 100 },
                        },
                        // Сортировка по времени по умолчанию (сначала новые)
                        sorting: {
                            sortModel: [{ field: 'timestamp', sort: 'desc' }],
                        },
                    }}
                    pageSizeOptions={[25, 50, 100, 200]}
                    disableRowSelectionOnClick
                    autoHeight={false}
                    sx={{
                         border: 0,
                         '& .MuiDataGrid-columnHeaders': { backgroundColor: 'action.hover' }
                    }}
                    // Может быть полезно для поля details
                    // getRowHeight={() => 'auto'}
                    // sx={{ ..., '&.MuiDataGrid-root--autoHeight .MuiDataGrid-cell': { overflow: 'visible', whiteSpace: 'normal !important' } }}
                />
            </Paper>
        </Box>
    );
}

export default ActivitiesPage; 
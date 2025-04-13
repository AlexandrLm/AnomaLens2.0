import React, { useState } from 'react';
import axios from 'axios'; // Импортируем axios
import {
    Typography, Box, TextField, Button, Grid, CircularProgress, Alert, Paper
} from '@mui/material';

function SimulatorPage() {
  // Состояние для полей формы
  const [config, setConfig] = useState({
    num_customers: 100,
    num_products: 50,
    num_orders: 200,
    num_activities: 1000,
    order_anomaly_rate: 0.05,
    activity_anomaly_rate: 0.05,
  });
  // Состояние для процесса отправки
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [summary, setSummary] = useState(null); // Для хранения сводки

  // Обработчик изменения полей формы
  const handleChange = (event) => {
    const { name, value, type } = event.target;
    // Преобразуем в число или float, если нужно
    const parsedValue = type === 'number' ? parseFloat(value) : value;
    setConfig(prevConfig => ({
      ...prevConfig,
      [name]: parsedValue,
    }));
  };

  // Обработчик отправки формы
  const handleSubmit = async (event) => {
    event.preventDefault(); // Предотвращаем стандартную отправку формы
    setLoading(true);
    setError('');
    setSummary(null);

    // Валидация (простая)
    if (config.order_anomaly_rate < 0 || config.order_anomaly_rate > 1 ||
        config.activity_anomaly_rate < 0 || config.activity_anomaly_rate > 1) {
        setError('Доля аномалий должна быть между 0.0 и 1.0');
        setLoading(false);
        return;
    }

    try {
      // Оборачиваем config в соответствии с ожидаемым форматом API { "config": { ... } }
      const requestBody = { config: config };
      console.log('Sending generation request:', requestBody); // Логируем запрос

      // Используем относительный путь /api/..., т.к. настроен proxy
      const response = await axios.post('/api/simulator/generate', requestBody);

      console.log('Generation response:', response.data); // Логируем ответ
      setSummary(response.data); // Сохраняем сводку из ответа
    } catch (err) {
      console.error("Generation failed:", err);
      let errorMessage = 'Ошибка при генерации данных.';
      if (err.response && err.response.data && err.response.data.detail) {
          errorMessage += ` Детали: ${err.response.data.detail}`;
      } else if (err.message) {
           errorMessage += ` ${err.message}`;
      }
      setError(errorMessage);
    } finally {
      setLoading(false); // Убираем индикатор загрузки
    }
  };

  return (
    <Paper elevation={3} sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Симулятор Данных
      </Typography>
      <Typography paragraph>
        Настройте параметры и запустите генерацию тестовых данных для базы.
        Генерация может занять некоторое время.
      </Typography>
      <Box component="form" onSubmit={handleSubmit} noValidate sx={{ mt: 1 }}>
        <Grid container spacing={2}>
          {/* Используем Grid для расположения полей */}
          <Grid item xs={12} sm={6} md={4}>
            <TextField
              margin="normal"
              required
              fullWidth
              id="num_customers"
              label="Кол-во клиентов"
              name="num_customers"
              type="number"
              value={config.num_customers}
              onChange={handleChange}
              disabled={loading}
              inputProps={{ min: 0 }}
            />
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <TextField
              margin="normal"
              required
              fullWidth
              id="num_products"
              label="Кол-во товаров"
              name="num_products"
              type="number"
              value={config.num_products}
              onChange={handleChange}
              disabled={loading}
               inputProps={{ min: 0 }}
           />
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
             <TextField
              margin="normal"
              required
              fullWidth
              id="num_orders"
              label="Кол-во заказов"
              name="num_orders"
              type="number"
              value={config.num_orders}
              onChange={handleChange}
              disabled={loading}
               inputProps={{ min: 0 }}
           />
          </Grid>
           <Grid item xs={12} sm={6} md={4}>
             <TextField
              margin="normal"
              required
              fullWidth
              id="num_activities"
              label="Кол-во активностей"
              name="num_activities"
              type="number"
              value={config.num_activities}
              onChange={handleChange}
              disabled={loading}
               inputProps={{ min: 0 }}
           />
          </Grid>
           <Grid item xs={12} sm={6} md={4}>
            <TextField
              margin="normal"
              required
              fullWidth
              id="order_anomaly_rate"
              label="Доля аном. заказов (0.0-1.0)"
              name="order_anomaly_rate"
              type="number"
              value={config.order_anomaly_rate}
              onChange={handleChange}
              disabled={loading}
              inputProps={{ step: 0.01, min: 0, max: 1 }}
            />
          </Grid>
           <Grid item xs={12} sm={6} md={4}>
            <TextField
              margin="normal"
              required
              fullWidth
              id="activity_anomaly_rate"
              label="Доля аном. активностей (0.0-1.0)"
              name="activity_anomaly_rate"
              type="number"
              value={config.activity_anomaly_rate}
              onChange={handleChange}
              disabled={loading}
              inputProps={{ step: 0.01, min: 0, max: 1 }}
            />
          </Grid>
        </Grid>

        <Button
          type="submit"
          fullWidth
          variant="contained"
          sx={{ mt: 3, mb: 2 }}
          disabled={loading}
        >
          {loading ? <CircularProgress size={24} /> : 'Сгенерировать Данные'}
        </Button>

        {/* Отображение ошибки */}
        {error && (
          <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>
        )}

        {/* Отображение сводки */}
        {summary && (
          <Alert severity="success" sx={{ mt: 2 }}>
            <Typography variant="subtitle1">Генерация завершена:</Typography>
            <ul>
              {Object.entries(summary).map(([key, value]) => (
                <li key={key}>{`${key}: ${value}`}</li>
              ))}
            </ul>
          </Alert>
        )}
      </Box>
    </Paper>
  );
}

export default SimulatorPage; 
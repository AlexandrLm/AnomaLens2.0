import React, { useState } from 'react';
import axios from 'axios'; // Импортируем axios
import {
    Typography, Box, TextField, Button, Grid, CircularProgress, Alert, Paper,
    FormControlLabel, Checkbox, Slider, Divider
} from '@mui/material';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFnsV3'; // Используем V3 адаптер
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { DatePicker } from '@mui/x-date-pickers/DatePicker';
import { ru } from 'date-fns/locale/ru'; // Локаль для date-fns

// Вспомогательная функция для форматирования значения слайдера в процентах
function valuetext(value) {
  return `${(value * 100).toFixed(0)}%`;
}

function SimulatorPage() {
  // Состояние для полей формы (обновляем дефолты)
  const [config, setConfig] = useState({
    num_customers: 100,
    num_products: 50,
    num_orders: 200,
    num_activities: 1000,
    // --- Новые параметры аномалий --- 
    enable_order_amount_anomaly: true,
    order_amount_anomaly_rate: 0.03,
    enable_activity_burst_anomaly: true,
    activity_burst_anomaly_rate: 0.02,
    enable_failed_login_anomaly: true,
    failed_login_anomaly_rate: 0.01,
    // --- Добавляем дефолты для дат (null) --- 
    activity_start_date: null,
    activity_end_date: null,
    // ----------------------------------------
  });
  // Состояние для процесса отправки
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [summary, setSummary] = useState(null); // Для хранения сводки

  // Обработчик изменения полей формы (добавляем обработку Checkbox и Slider)
  const handleChange = (event, newValue) => { // newValue для Slider
    const { name, value, type, checked } = event.target;
    let parsedValue;

    if (type === 'checkbox') {
        parsedValue = checked;
    } else if (type === 'number') {
        parsedValue = parseFloat(value);
    } else { // Обработка Slider
        // Если newValue передано (из Slider), используем его
        // Иначе, если есть event.target.name, это может быть TextField (хотя Slider тоже имеет name)
        // Эта логика немного запутана, но должна работать для Slider и TextField
        if (newValue !== undefined) { 
             // Slider возвращает значение напрямую, name берем из самого Slider
             // Нужно найти имя слайдера по значению? Нет, Slider передает name в event.
             const sliderName = event.target.name || event.target.htmlFor; // Пытаемся получить имя
             if (sliderName) {
                 setConfig(prevConfig => ({
                    ...prevConfig,
                    [sliderName]: newValue,
                 }));
             }
             return; // Выходим, т.к. обработали Slider
        } else {
            parsedValue = value; // Для TextField (если type не number)
        }
    }
    
    // Установка состояния для Checkbox и TextField (не-слайдер)
    if (name) { // Убедимся, что имя есть
        setConfig(prevConfig => ({
            ...prevConfig,
            [name]: parsedValue,
        }));
    }
  };

  // Обработчик изменения дат
  const handleDateChange = (name, newValue) => {
    setConfig(prevConfig => ({
      ...prevConfig,
      [name]: newValue, // newValue будет объектом Date или null
    }));
  };

  // Обработчик отправки формы (убираем старую валидацию anomaly_rate)
  const handleSubmit = async (event) => {
    event.preventDefault(); // Предотвращаем стандартную отправку формы
    setLoading(true);
    setError('');
    setSummary(null);

    try {
      // --- Конвертируем даты в ISO строки или null --- 
      const configToSend = {
        ...config,
        activity_start_date: config.activity_start_date ? config.activity_start_date.toISOString() : null,
        activity_end_date: config.activity_end_date ? config.activity_end_date.toISOString() : null,
      };
      // ------------------------------------------
      
      // Оборачиваем configToSend в соответствии с ожидаемым форматом API
      const requestBody = { config: configToSend };
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
    <LocalizationProvider dateAdapter={AdapterDateFns} adapterLocale={ru}>
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
            {/* --- Добавляем Date Pickers --- */}
            <Grid item xs={12} sm={6}>
              <DatePicker
                label="Начальная дата активностей"
                value={config.activity_start_date}
                onChange={(newValue) => handleDateChange('activity_start_date', newValue)}
                slotProps={{ textField: { fullWidth: true, margin: 'normal' } }}
                disabled={loading}
                maxDate={config.activity_end_date || undefined} // Нельзя выбрать дату позже конечной
              />
            </Grid>
            <Grid item xs={12} sm={6}>
               <DatePicker
                label="Конечная дата активностей"
                value={config.activity_end_date}
                onChange={(newValue) => handleDateChange('activity_end_date', newValue)}
                slotProps={{ textField: { fullWidth: true, margin: 'normal' } }}
                disabled={loading}
                minDate={config.activity_start_date || undefined} // Нельзя выбрать дату раньше начальной
              />
            </Grid>
          </Grid>

          {/* --- Новая секция для настройки аномалий --- */}
          <Divider sx={{ my: 3 }}><Typography variant="overline">Настройки Аномалий</Typography></Divider>
          
          <Grid container spacing={2} alignItems="center">
              {/* Аномалии Суммы Заказа */}
              <Grid item xs={12} md={4}>
                  <FormControlLabel 
                      control={<Checkbox checked={config.enable_order_amount_anomaly} onChange={handleChange} name="enable_order_amount_anomaly" disabled={loading} />} 
                      label="Аномалии Суммы Заказа"
                  />
              </Grid>
              <Grid item xs={12} md={8}>
                   <Typography gutterBottom variant="caption">Доля аном. заказов:</Typography>
                   <Slider
                      name="order_amount_anomaly_rate"
                      value={config.order_amount_anomaly_rate}
                      onChange={handleChange}
                      valueLabelDisplay="auto"
                      getAriaValueText={valuetext}
                      step={0.01}
                      marks
                      min={0}
                      max={0.2} // Ограничим 20%, чтобы не было слишком много
                      disabled={loading || !config.enable_order_amount_anomaly}
                      valueLabelFormat={valuetext} // Показываем % во всплывающей подсказке
                  />
              </Grid>

              {/* Аномалии Burst Активности */}
              <Grid item xs={12} md={4}>
                  <FormControlLabel 
                      control={<Checkbox checked={config.enable_activity_burst_anomaly} onChange={handleChange} name="enable_activity_burst_anomaly" disabled={loading} />} 
                      label="Аномалии Burst Активности"
                  />
              </Grid>
              <Grid item xs={12} md={8}>
                   <Typography gutterBottom variant="caption">Шанс Burst-сессии:</Typography>
                   <Slider
                      name="activity_burst_anomaly_rate"
                      value={config.activity_burst_anomaly_rate}
                      onChange={handleChange}
                      valueLabelDisplay="auto"
                      getAriaValueText={valuetext}
                      step={0.01}
                      marks
                      min={0}
                      max={0.15} // Ограничим 15%
                      disabled={loading || !config.enable_activity_burst_anomaly}
                      valueLabelFormat={valuetext}
                  />
              </Grid>

              {/* Аномалии Failed Login */}
              <Grid item xs={12} md={4}>
                  <FormControlLabel 
                      control={<Checkbox checked={config.enable_failed_login_anomaly} onChange={handleChange} name="enable_failed_login_anomaly" disabled={loading} />} 
                      label="Аномалии Failed Login"
                  />
              </Grid>
              <Grid item xs={12} md={8}>
                   <Typography gutterBottom variant="caption">Шанс Failed Login сессии:</Typography>
                   <Slider
                      name="failed_login_anomaly_rate"
                      value={config.failed_login_anomaly_rate}
                      onChange={handleChange}
                      valueLabelDisplay="auto"
                      getAriaValueText={valuetext}
                      step={0.005}
                      marks
                      min={0}
                      max={0.1} // Ограничим 10%
                      disabled={loading || !config.enable_failed_login_anomaly}
                      valueLabelFormat={valuetext}
                  />
              </Grid>
          </Grid>
          {/* ---------------------------------------- */}

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
    </LocalizationProvider>
  );
}

export default SimulatorPage; 
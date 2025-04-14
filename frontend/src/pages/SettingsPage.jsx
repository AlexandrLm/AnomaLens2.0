import React, { useState, useEffect, useCallback } from 'react';
// Убираем axios и связанные state
// import axios from 'axios';
import {
    Box, Typography, TextField, Button, Paper, Grid,
    Card,
    CardContent,
    Divider,
    IconButton,
    Tooltip,
    Zoom,
    alpha,
    InputAdornment,
    Fade
} from '@mui/material';
import {
    Save as SaveIcon,
    Refresh as RefreshIcon,
    Info as InfoIcon,
    HelpOutline as HelpIcon
} from '@mui/icons-material';
import Switch from '@mui/material/Switch';
import FormControlLabel from '@mui/material/FormControlLabel';

// Ключи настроек и дефолтные значения (ВОЗВРАЩАЕМ)
const SETTING_KEYS = [
    'limit', 
    'z_threshold', 
    'dbscan_eps', 
    'dbscan_min_samples',
    'use_autoencoder',
    'autoencoder_threshold'
];
const defaultSettings = {
    limit: 10000,
    z_threshold: 3.0,
    dbscan_eps: 0.5,
    dbscan_min_samples: 5,
    use_autoencoder: false,
    autoencoder_threshold: 0.5,
};

// Описания настроек
const settingDescriptions = {
    limit: "Максимальное количество записей для анализа и обучения. Влияет на производительность и точность.",
    z_threshold: "Пороговое значение z-оценки для статистического детектора. Чем выше, тем меньше аномалий будет найдено.",
    dbscan_eps: "Размер окрестности точки для алгоритма DBSCAN. Влияет на определение кластеров.",
    dbscan_min_samples: "Минимальное количество точек для формирования кластера в DBSCAN. Влияет на чувствительность алгоритма.",
    use_autoencoder: "Включить/выключить использование детектора Autoencoder.",
    autoencoder_threshold: "Порог ошибки реконструкции (MSE) для Autoencoder. Значения выше считаются аномалиями."
};

// Функция для форматирования названий настроек
const formatSettingName = (key) => {
    if (key === 'use_autoencoder') return 'Use Autoencoder';
    if (key === 'autoencoder_threshold') return 'Autoencoder Threshold (MSE)';
    return key
        .replace(/_/g, ' ')
        .split(' ')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
};

function SettingsPage() {
    // ВОЗВРАЩАЕМ ПРОСТОЙ STATE И ЛОГИКУ localStorage
    const [settings, setSettings] = useState(() => {
        const savedSettings = localStorage.getItem('anomalySettings');
        if (savedSettings) {
            try {
                const parsed = JSON.parse(savedSettings);
                const completeSettings = { ...defaultSettings };
                for (const key of SETTING_KEYS) {
                    if (parsed[key] !== undefined) {
                        if (key === 'limit' || key === 'dbscan_min_samples') {
                            completeSettings[key] = parseInt(parsed[key], 10) || defaultSettings[key];
                        } else if (key === 'z_threshold' || key === 'dbscan_eps' || key === 'autoencoder_threshold') {
                            completeSettings[key] = parseFloat(parsed[key]) || defaultSettings[key];
                        } else if (key === 'use_autoencoder') {
                            completeSettings[key] = typeof parsed[key] === 'boolean' ? parsed[key] : defaultSettings[key];
                        } else {
                            completeSettings[key] = parsed[key];
                        }
                    } else {
                        completeSettings[key] = defaultSettings[key];
                    }
                }
                return completeSettings;
            } catch (error) {
                console.error("Failed to parse settings from localStorage on init:", error);
                return defaultSettings; // Ошибка парсинга, используем дефолт
            }
        }
        return defaultSettings; // Нет сохраненных, используем дефолт
    });
    const [statusMessage, setStatusMessage] = useState('');
    const [savedAnimation, setSavedAnimation] = useState(false);

    // УДАЛЯЕМ useEffect для загрузки из API
    // useEffect(() => { ... });

    // ВОЗВРАЩАЕМ ОБРАБОТЧИК ИЗМЕНЕНИЙ ДЛЯ ЧИСЕЛ
    const handleChange = (event) => {
        const { name, value, type, checked } = event.target;
        
        if (type === 'checkbox') {
            setSettings(prev => ({ ...prev, [name]: checked }));
        } else {
            if (value === '' || (value === '-' && (name === 'z_threshold' || name === 'dbscan_eps' || name === 'autoencoder_threshold'))) {
                 setSettings(prev => ({ ...prev, [name]: value }));
                 return;
            }
            const numValue = (name === 'limit' || name === 'dbscan_min_samples') 
                           ? parseInt(value, 10) 
                           : parseFloat(value);

            if (!isNaN(numValue)) {
                setSettings(prev => ({ ...prev, [name]: numValue }));
            } else {
                console.warn(`Invalid input for ${name}: ${value}`);
            }
        }
    };

     // ВОЗВРАЩАЕМ ОБРАБОТЧИК ПОТЕРИ ФОКУСА ДЛЯ ВАЛИДАЦИИ
    const handleBlur = (event) => {
        const { name, value, type } = event.target;
        if (type === 'checkbox') return;

        let finalValue = settings[name];
        
        if (value === '' || (typeof finalValue !== 'number' && name !== 'use_autoencoder') || (typeof finalValue === 'number' && isNaN(finalValue)) ) {
             finalValue = defaultSettings[name];
        } else if (typeof finalValue === 'number') {
             if ((name === 'limit' || name === 'dbscan_min_samples') && finalValue < 1) {
                 finalValue = defaultSettings[name];
             } else if ((name === 'z_threshold' || name === 'dbscan_eps' || name === 'autoencoder_threshold') && finalValue < 0) {
                  finalValue = defaultSettings[name];
             }
        }
        if (settings[name] !== finalValue) { 
            setSettings(prev => ({ ...prev, [name]: finalValue }));
        }
    };

    // ВОЗВРАЩАЕМ ОБРАБОТЧИК СОХРАНЕНИЯ В localStorage
    const handleSave = () => {
        console.log("Saving settings to localStorage:", settings);
         const finalSettings = { ...settings };
         let reverted = false;
         for (const key of SETTING_KEYS) {
             const expectedType = typeof defaultSettings[key];
             if (expectedType === 'number' && (typeof finalSettings[key] !== 'number' || isNaN(finalSettings[key]))) {
                 finalSettings[key] = defaultSettings[key];
                 reverted = true;
             } else if (expectedType === 'boolean' && typeof finalSettings[key] !== 'boolean') {
                 finalSettings[key] = defaultSettings[key];
                 reverted = true;
             }
             if (expectedType === 'number') {
                 if ((key === 'limit' || key === 'dbscan_min_samples') && finalSettings[key] < 1) {
                     finalSettings[key] = defaultSettings[key]; reverted = true;
                 } else if ((key === 'z_threshold' || key === 'dbscan_eps' || key === 'autoencoder_threshold') && finalSettings[key] < 0) {
                     finalSettings[key] = defaultSettings[key]; reverted = true;
                 }
             }
         }
         if (reverted) {
             setSettings(finalSettings);
             console.warn("Reverted some invalid settings to defaults before saving.");
         }
        try {
            localStorage.setItem('anomalySettings', JSON.stringify(finalSettings));
            setStatusMessage('Настройки успешно сохранены!');
            setSavedAnimation(true);
            setTimeout(() => { setStatusMessage(''); setSavedAnimation(false); }, 3000);
        } catch (error) {
            console.error("Error saving settings to localStorage:", error);
            setStatusMessage('Ошибка сохранения настроек!');
        }
    };

    const handleReset = () => {
        setSettings(defaultSettings);
        setStatusMessage('Настройки сброшены до значений по умолчанию');
        setTimeout(() => setStatusMessage(''), 3000);
    };

    // УДАЛЯЕМ обработчик закрытия Snackbar
    // const handleCloseSnackbar = (...) => { ... };

    // УДАЛЯЕМ логику загрузки/ошибки API
    // if (initialLoading) { ... }
    // if (loadingError) { ... }

    // ВОЗВРАЩАЕМ старый рендер
    return (
        <Box sx={{ padding: 3 }}>
            <Box 
                sx={{ 
                    mb: 4, 
                    display: 'flex', 
                    alignItems: 'center',
                    position: 'relative',
                    '&::after': {
                        content: '""',
                        position: 'absolute',
                        bottom: -10,
                        left: 0,
                        width: '60px',
                        height: '4px',
                        borderRadius: '2px',
                        background: (theme) => `linear-gradient(90deg, ${theme.palette.primary.main}, ${theme.palette.primary.light})`
                    }
                }}
            >
                <Typography 
                    variant="h4" 
                    sx={{ 
                        fontWeight: 700,
                        background: (theme) => `linear-gradient(90deg, ${theme.palette.primary.dark}, ${theme.palette.primary.main})`,
                        WebkitBackgroundClip: 'text',
                        WebkitTextFillColor: 'transparent',
                        letterSpacing: '0.5px'
                    }}
                >
                    Настройки Обнаружения Аномалий
                </Typography>
            </Box>
            
            <Grid container spacing={3}>
                <Grid item xs={12} md={8}>
                    <Fade in={true} timeout={800}>
                        <Card 
                            elevation={3} 
                            sx={{ 
                                borderRadius: 3,
                                background: 'white',
                                position: 'relative',
                                overflow: 'visible',
                                '&::before': {
                                    content: '""',
                                    position: 'absolute',
                                    top: 0,
                                    left: 0,
                                    right: 0,
                                    height: '6px',
                                    background: (theme) => `linear-gradient(90deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
                                    borderTopLeftRadius: '16px',
                                    borderTopRightRadius: '16px'
                                }
                            }}
                        >
                            <CardContent sx={{ p: 4 }}>
                                <Typography variant="h6" sx={{ mb: 3, fontWeight: 600 }}>
                                    Параметры Алгоритмов
                                </Typography>
                                
                                <Grid container spacing={3}>
                                    {SETTING_KEYS.map((key, index) => {
                                        const isBooleanField = key === 'use_autoencoder';
                                        const isNumberField = typeof defaultSettings[key] === 'number';

                                        return (
                                            <Zoom 
                                                in={true} 
                                                style={{ 
                                                    transitionDelay: `${index * 100}ms`
                                                }}
                                                key={key}
                                            >
                                                <Grid item xs={12} sm={isBooleanField ? 12 : 6}>
                                                    {isBooleanField ? (
                                                        <FormControlLabel
                                                            control={
                                                                <Switch
                                                                    checked={settings[key]}
                                                                    onChange={handleChange}
                                                                    name={key}
                                                                    color="primary"
                                                                />
                                                            }
                                                            label={
                                                                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                                                    {formatSettingName(key)}
                                                                    <Tooltip title={settingDescriptions[key]} placement="top" arrow>
                                                                        <HelpIcon sx={{ ml: 1, fontSize: '1rem', color: 'text.secondary' }} />
                                                                    </Tooltip>
                                                                </Box>
                                                            }
                                                            sx={{ width: '100%', justifyContent: 'space-between', ml: 0, mt: 2, mb: 1 }}
                                                        />
                                                    ) : (
                                                        <TextField
                                                            fullWidth
                                                            variant="outlined"
                                                            label={formatSettingName(key)}
                                                            type={isNumberField ? "number" : "text"}
                                                            name={key}
                                                            value={settings[key]}
                                                            onChange={handleChange}
                                                            onBlur={handleBlur}
                                                            InputProps={{
                                                                inputProps: {
                                                                    step: (key === 'limit' || key === 'dbscan_min_samples') ? 1 : 0.01,
                                                                    min: (key === 'limit' || key === 'dbscan_min_samples') ? 1 : 0,
                                                                },
                                                                endAdornment: (
                                                                    <InputAdornment position="end">
                                                                        <Tooltip title={settingDescriptions[key]} placement="top" arrow>
                                                                            <HelpIcon sx={{ fontSize: '1rem', color: 'text.secondary' }} />
                                                                        </Tooltip>
                                                                    </InputAdornment>
                                                                ),
                                                            }}
                                                            sx={{
                                                                '& .MuiOutlinedInput-root': {
                                                                    backgroundColor: (theme) => alpha(theme.palette.background.paper, 0.8),
                                                                    transition: 'all 0.3s ease',
                                                                    '&:hover': {
                                                                        backgroundColor: 'white',
                                                                    },
                                                                    '&.Mui-focused': {
                                                                        backgroundColor: 'white',
                                                                    }
                                                                }
                                                            }}
                                                        />
                                                    )}
                                                </Grid>
                                            </Zoom>
                                        );
                                    })}
                                </Grid>
                                
                                <Divider sx={{ my: 4 }} />
                                
                                <Box sx={{ 
                                    display: 'flex', 
                                    justifyContent: 'space-between',
                                    alignItems: 'center',
                                }}>
                                    <Box sx={{ display: 'flex', gap: 2 }}>
                                        <Button
                                            variant="contained"
                                            color="primary"
                                            onClick={handleSave}
                                            startIcon={<SaveIcon />}
                                            sx={{ 
                                                px: 3, 
                                                position: 'relative',
                                                overflow: 'hidden',
                                                '&::after': savedAnimation ? {
                                                    content: '""',
                                                    position: 'absolute',
                                                    width: '100%',
                                                    height: '100%',
                                                    top: 0,
                                                    left: 0,
                                                    background: 'radial-gradient(circle, rgba(255,255,255,0.8) 0%, rgba(255,255,255,0) 70%)',
                                                    animation: 'ripple 1s ease-out forwards'
                                                } : {}
                                            }}
                                        >
                                            Сохранить
                                        </Button>
                                        
                                        <Button
                                            variant="outlined"
                                            color="secondary"
                                            onClick={handleReset}
                                            startIcon={<RefreshIcon />}
                                        >
                                            Сбросить
                                        </Button>
                                    </Box>
                                    
                                    {statusMessage && (
                                        <Fade in={!!statusMessage}>
                                            <Typography 
                                                sx={{ 
                                                    color: statusMessage.startsWith('Ошибка') 
                                                        ? 'error.main' 
                                                        : 'success.main',
                                                    fontWeight: 500,
                                                    display: 'flex',
                                                    alignItems: 'center',
                                                    gap: 1,
                                                    px: 2,
                                                    py: 1,
                                                    borderRadius: 2,
                                                    backgroundColor: statusMessage.startsWith('Ошибка')
                                                        ? (theme) => alpha(theme.palette.error.main, 0.1)
                                                        : (theme) => alpha(theme.palette.success.main, 0.1),
                                                }}
                                            >
                                                {statusMessage.startsWith('Ошибка')
                                                    ? <InfoIcon color="error" fontSize="small" />
                                                    : <SaveIcon color="success" fontSize="small" />
                                                }
                                                {statusMessage}
                                            </Typography>
                                        </Fade>
                                    )}
                                </Box>
                            </CardContent>
                        </Card>
                    </Fade>
                </Grid>
                
                <Grid item xs={12} md={4}>
                    <Fade in={true} timeout={1000}>
                        <Card 
                            elevation={2} 
                            sx={{ 
                                height: '100%', 
                                display: 'flex', 
                                flexDirection: 'column',
                                borderRadius: 3,
                                background: (theme) => `linear-gradient(145deg, ${alpha(theme.palette.info.light, 0.1)} 0%, ${alpha(theme.palette.primary.light, 0.1)} 100%)`,
                                border: (theme) => `1px solid ${alpha(theme.palette.primary.main, 0.1)}`,
                            }}
                        >
                            <CardContent sx={{ p: 3, flexGrow: 1 }}>
                                <Box 
                                    sx={{ 
                                        display: 'flex', 
                                        alignItems: 'center', 
                                        mb: 3,
                                        gap: 1
                                    }}
                                >
                                    <InfoIcon 
                                        color="info" 
                                        fontSize="small"
                                        sx={{
                                            backgroundColor: (theme) => alpha(theme.palette.info.main, 0.1),
                                            padding: 0.5,
                                            borderRadius: '50%'
                                        }}
                                    />
                                    <Typography variant="h6" color="info.main" fontWeight={600}>
                                        Информация о Настройках
                                    </Typography>
                                </Box>
                                
                                <Typography variant="body1" paragraph sx={{ opacity: 0.9 }}>
                                    Настройки хранятся локально в вашем браузере и применяются ко всем операциям обнаружения аномалий.
                                </Typography>
                                
                                <Typography variant="body1" paragraph sx={{ opacity: 0.9 }}>
                                    Изменение параметров влияет на чувствительность алгоритмов и может существенно повлиять на количество обнаруживаемых аномалий.
                                </Typography>
                                
                                <Box 
                                    sx={{ 
                                        mt: 3, 
                                        p: 2, 
                                        borderRadius: 2, 
                                        bgcolor: 'background.paper',
                                        border: (theme) => `1px dashed ${theme.palette.divider}`
                                    }}
                                >
                                    <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 600 }}>
                                        Рекомендации по настройке:
                                    </Typography>
                                    <Typography variant="body2" sx={{ fontSize: '0.85rem', opacity: 0.85 }}>
                                        • Увеличьте <b>Z Threshold</b> для уменьшения количества выявляемых аномалий<br />
                                        • Уменьшите <b>DBSCAN Eps</b> для обнаружения более плотных кластеров<br />
                                        • Увеличьте <b>Limit</b> для более точного анализа (может снизить производительность)
                                    </Typography>
                                </Box>
                            </CardContent>
                        </Card>
                    </Fade>
                </Grid>
            </Grid>
            
            <style jsx global>{`
                @keyframes ripple {
                    from {
                        opacity: 1;
                        transform: scale(0);
                    }
                    to {
                        opacity: 0;
                        transform: scale(2);
                    }
                }
            `}</style>
        </Box>
    );
}

// ОСТАВЛЯЕМ ТОЛЬКО ОДИН ЭКСПОРТ
export default SettingsPage; 
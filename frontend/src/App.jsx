import React, { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom';
import { 
  Box, 
  CssBaseline, 
  AppBar, 
  Toolbar, 
  Drawer, 
  List, 
  ListItem, 
  ListItemIcon, 
  ListItemText, 
  Typography, 
  useTheme, 
  Divider, 
  Avatar, 
  IconButton,
  Tooltip,
  useMediaQuery,
  Fade
} from '@mui/material';
import { ThemeProvider } from '@mui/material/styles';
import { 
  Dashboard, 
  Timeline, 
  Settings as SettingsIcon, 
  Group, 
  ShoppingCart, 
  Category, 
  ReceiptLong, 
  Menu as MenuIcon,
  ChevronLeft,
  Brightness4,
  Brightness7,
  SearchRounded,
  NotificationsNone,
  FiberManualRecord,
  Info
} from '@mui/icons-material';

import DashboardPage from './pages/DashboardPage';
import SimulatorPage from './pages/SimulatorPage';
import SettingsPage from './pages/SettingsPage';
import CustomersPage from './pages/CustomersPage';
import ProductsPage from './pages/ProductsPage';
import OrdersPage from './pages/OrdersPage';
import ActivitiesPage from './pages/ActivitiesPage';
import ExplanationPage from './pages/ExplanationPage';
import theme from './theme';

const drawerWidth = 280;

const menuItems = [
  { text: 'Панель Управления', icon: <Dashboard />, path: '/' },
  { text: 'Симулятор Данных', icon: <Timeline />, path: '/simulator' },
  { divider: true },
  { text: 'Пользователи', icon: <Group />, path: '/customers' },
  { text: 'Продукты', icon: <Category />, path: '/products' },
  { text: 'Заказы', icon: <ShoppingCart />, path: '/orders' },
  { text: 'Активности', icon: <ReceiptLong />, path: '/activities' },
  { divider: true },
  { text: 'О Методах', icon: <Info />, path: '/explanation' },
  { text: 'Настройки', icon: <SettingsIcon />, path: '/settings' },
];

function App() {
  const [open, setOpen] = useState(true);
  const isSmallScreen = useMediaQuery(theme.breakpoints.down('md'));
  
  useEffect(() => {
    // Close drawer by default on small screens
    if (isSmallScreen) {
      setOpen(false);
    } else {
      setOpen(true);
    }
  }, [isSmallScreen]);

  const toggleDrawer = () => {
    setOpen(!open);
  };

  return (
    <ThemeProvider theme={theme}>
      <BrowserRouter>
        <Box sx={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
          <CssBaseline />
          <AppBar 
            position="fixed" 
            sx={{ 
              zIndex: theme.zIndex.drawer + 1,
              transition: 'width 0.25s',
              width: { sm: open ? `calc(100% - ${drawerWidth}px)` : '100%' },
              ml: { sm: open ? `${drawerWidth}px` : 0 },
            }}
          >
            <Toolbar sx={{ justifyContent: 'space-between' }}>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <IconButton
                  color="inherit"
                  aria-label="open drawer"
                  edge="start"
                  onClick={toggleDrawer}
                  sx={{ mr: 2, display: 'flex' }}
                >
                  {open ? <ChevronLeft /> : <MenuIcon />}
                </IconButton>
                <Typography variant="h6" noWrap component="div" sx={{ fontWeight: 700, letterSpacing: '0.5px' }}>
                  AnomaLens 2.0
                </Typography>
              </Box>
              
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Tooltip title="Поиск">
                  <IconButton color="inherit" sx={{ borderRadius: 2, bgcolor: 'rgba(255,255,255,0.1)', '&:hover': { bgcolor: 'rgba(255,255,255,0.2)' } }}>
                    <SearchRounded />
                  </IconButton>
                </Tooltip>
                <Tooltip title="Уведомления">
                  <IconButton color="inherit" sx={{ 
                    borderRadius: 2, 
                    bgcolor: 'rgba(255,255,255,0.1)', 
                    '&:hover': { bgcolor: 'rgba(255,255,255,0.2)' },
                    position: 'relative'
                  }}>
                    <NotificationsNone />
                    <Box 
                      component="span" 
                      sx={{ 
                        position: 'absolute', 
                        top: 10, 
                        right: 10, 
                        width: 8, 
                        height: 8, 
                        borderRadius: '50%', 
                        bgcolor: theme.palette.error.main,
                        animation: 'pulse 1.5s infinite'
                      }}
                    />
                  </IconButton>
                </Tooltip>
                <Tooltip title="Администратор">
                  <Avatar 
                    sx={{ 
                      width: 38, 
                      height: 38, 
                      cursor: 'pointer',
                      bgcolor: theme.palette.primary.main,
                      border: '2px solid rgba(255,255,255,0.2)',
                      transition: 'transform 0.2s',
                      '&:hover': {
                        transform: 'scale(1.1)'
                      }
                    }}
                  >A</Avatar>
                </Tooltip>
              </Box>
            </Toolbar>
          </AppBar>
          
          <Drawer
            variant={isSmallScreen ? "temporary" : "persistent"}
            open={open}
            onClose={toggleDrawer}
            sx={{
              width: drawerWidth,
              flexShrink: 0,
              [`& .MuiDrawer-paper`]: { 
                width: drawerWidth, 
                boxSizing: 'border-box',
                borderRight: '1px solid rgba(0,0,0,0.05)',
                background: 'linear-gradient(180deg, rgba(255,255,255,1) 0%, rgba(249,250,251,1) 100%)',
              },
            }}
          >
            <Toolbar 
              sx={{ 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'center',
                py: 2
              }}
            >
              <Box 
                component="img"
                src="/logo.svg"
                alt="AnomaLens Logo"
                sx={{ 
                  height: 40,
                  width: 'auto',
                  display: 'block',
                  opacity: 0.9
                }}
              />
            </Toolbar>
            <Divider sx={{ mx: 2 }} />
            <Box sx={{ overflow: 'auto', py: 2, height: '100%' }}>
              <List sx={{ px: 2 }}>
                {menuItems.map((item, index) => (
                  item.divider ? (
                    <Divider key={`divider-${index}`} sx={{ my: 2 }} />
                  ) : (
                    <Fade key={item.text} in={true} style={{ transitionDelay: `${index * 50}ms` }}>
                      <ListItem 
                        button 
                        component={NavLink} 
                        to={item.path}
                        sx={{
                          mb: 1,
                          borderRadius: 2,
                          overflow: 'hidden',
                          '&.active': {
                            bgcolor: 'primary.main',
                            color: 'primary.contrastText',
                            boxShadow: '0 4px 12px rgba(51, 102, 255, 0.15)',
                            '& .MuiListItemIcon-root': {
                              color: 'primary.contrastText',
                            },
                            '&::before': {
                              content: '""',
                              position: 'absolute',
                              width: '6px',
                              height: '75%',
                              bgcolor: 'white',
                              borderRadius: '3px',
                              left: '6px',
                            }
                          },
                          transition: 'all 0.2s ease-in-out',
                          '&:hover': {
                            transform: 'translateX(4px)',
                            bgcolor: 'rgba(51, 102, 255, 0.08)',
                          }
                        }}
                      >
                        <ListItemIcon sx={{ 
                          minWidth: 40, 
                          color: 'grey.600',
                          transition: 'all 0.2s'
                        }}>
                          {item.icon}
                        </ListItemIcon>
                        <ListItemText 
                          primary={item.text} 
                          primaryTypographyProps={{ 
                            fontWeight: 500,
                            fontSize: '0.95rem',
                          }} 
                        />
                      </ListItem>
                    </Fade>
                  )
                ))}
              </List>
            </Box>
          </Drawer>
          
          <Box 
            component="main" 
            sx={{ 
              flexGrow: 1, 
              p: 3, 
              transition: 'margin 0.25s',
              marginLeft: open ? 0 : `-${drawerWidth}px`,
              width: '100%',
              height: '100%',
              overflow: 'auto',
              bgcolor: 'background.default',
              pt: '84px', // Extra padding for toolbar
            }}
          >
            <Routes>
              <Route path="/" element={<DashboardPage />} />
              <Route path="/simulator" element={<SimulatorPage />} />
              <Route path="/settings" element={<SettingsPage />} />
              <Route path="/customers" element={<CustomersPage />} />
              <Route path="/products" element={<ProductsPage />} />
              <Route path="/orders" element={<OrdersPage />} />
              <Route path="/activities" element={<ActivitiesPage />} />
              <Route path="/explanation" element={<ExplanationPage />} />
            </Routes>
          </Box>
        </Box>
      </BrowserRouter>
    </ThemeProvider>
  );
}

export default App;

import { createTheme, alpha } from '@mui/material/styles';

// Modern color palette with cohesive hues
const primaryMain = '#3366FF';
const secondaryMain = '#34C0AC';
const errorMain = '#FF5A5F';
const warningMain = '#FF9F45'; 
const infoMain = '#0EA5E9';
const successMain = '#10B981';
const neutralDark = '#111827';
const neutralLight = '#F3F4F6';

// Create a custom theme
const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: primaryMain,
      light: alpha(primaryMain, 0.8),
      dark: '#2952CC',
      contrastText: '#FFFFFF',
    },
    secondary: {
      main: secondaryMain,
      light: alpha(secondaryMain, 0.8),
      dark: '#2A9A8A',
      contrastText: '#FFFFFF',
    },
    error: {
      main: errorMain,
      light: alpha(errorMain, 0.8),
      dark: '#CC484C',
      contrastText: '#FFFFFF',
    },
    warning: {
      main: warningMain,
      light: alpha(warningMain, 0.8),
      dark: '#CC7F38',
      contrastText: '#FFFFFF',
    },
    info: {
      main: infoMain,
      light: alpha(infoMain, 0.8),
      dark: '#0B84BA',
      contrastText: '#FFFFFF',
    },
    success: {
      main: successMain,
      light: alpha(successMain, 0.8),
      dark: '#0D9467',
      contrastText: '#FFFFFF',
    },
    grey: {
      50: '#F9FAFB',
      100: '#F3F4F6',
      200: '#E5E7EB',
      300: '#D1D5DB',
      400: '#9CA3AF',
      500: '#6B7280',
      600: '#4B5563',
      700: '#374151',
      800: '#1F2937',
      900: '#111827',
    },
    background: {
      default: '#F9FAFB',
      paper: '#FFFFFF',
    },
    text: {
      primary: neutralDark,
      secondary: '#4B5563',
      disabled: '#9CA3AF',
    },
    divider: '#E5E7EB',
  },
  typography: {
    fontFamily: '"Inter", "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
    h1: {
      fontWeight: 700,
      fontSize: '2.5rem',
      lineHeight: 1.2,
      letterSpacing: '-0.01562em',
    },
    h2: {
      fontWeight: 700,
      fontSize: '2rem',
      lineHeight: 1.2,
      letterSpacing: '-0.00833em',
    },
    h3: {
      fontWeight: 600,
      fontSize: '1.75rem',
      lineHeight: 1.2,
      letterSpacing: '0em',
    },
    h4: {
      fontWeight: 600,
      fontSize: '1.5rem',
      lineHeight: 1.2,
      letterSpacing: '0.00735em',
    },
    h5: {
      fontWeight: 600,
      fontSize: '1.25rem',
      lineHeight: 1.2,
      letterSpacing: '0em',
    },
    h6: {
      fontWeight: 600,
      fontSize: '1rem',
      lineHeight: 1.2,
      letterSpacing: '0.0075em',
    },
    subtitle1: {
      fontWeight: 500,
      fontSize: '1rem',
      lineHeight: 1.5,
      letterSpacing: '0.00938em',
    },
    subtitle2: {
      fontWeight: 500,
      fontSize: '0.875rem',
      lineHeight: 1.57,
      letterSpacing: '0.00714em',
    },
    body1: {
      fontWeight: 400,
      fontSize: '1rem',
      lineHeight: 1.5,
      letterSpacing: '0.00938em',
    },
    body2: {
      fontWeight: 400,
      fontSize: '0.875rem',
      lineHeight: 1.57,
      letterSpacing: '0.00714em',
    },
    button: {
      fontWeight: 500,
      fontSize: '0.875rem',
      lineHeight: 1.75,
      letterSpacing: '0.02857em',
      textTransform: 'none',
    },
  },
  shape: {
    borderRadius: 10,
  },
  shadows: [
    'none',
    '0px 2px 4px rgba(17, 24, 39, 0.06), 0px 1px 2px rgba(17, 24, 39, 0.04)',
    '0px 4px 6px -1px rgba(17, 24, 39, 0.1), 0px 2px 4px -1px rgba(17, 24, 39, 0.06)',
    '0px 10px 15px -3px rgba(17, 24, 39, 0.1), 0px 4px 6px -2px rgba(17, 24, 39, 0.05)',
    '0px 20px 25px -5px rgba(17, 24, 39, 0.1), 0px 10px 10px -5px rgba(17, 24, 39, 0.04)',
    // Keep the default for higher indices (5-24)
    ...Array(20).fill('').map((_, i) => i < 3 ? '' : createTheme().shadows[i + 2]),
  ],
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        '*': {
          boxSizing: 'border-box',
        },
        html: {
          margin: 0,
          padding: 0,
          width: '100%',
          height: '100%',
          WebkitOverflowScrolling: 'touch',
        },
        body: {
          margin: 0,
          padding: 0,
          width: '100%',
          height: '100%',
        },
        '#root': {
          width: '100%',
          height: '100%',
        },
        a: {
          textDecoration: 'none',
          color: primaryMain,
        },
        'input:-webkit-autofill': {
          WebkitBoxShadow: '0 0 0 100px #fff inset',
          WebkitTextFillColor: '#000',
        },
        'input::-webkit-outer-spin-button, input::-webkit-inner-spin-button': {
          WebkitAppearance: 'none',
          margin: 0,
        },
        'input[type=number]': {
          MozAppearance: 'textfield',
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          background: `linear-gradient(90deg, ${primaryMain} 0%, ${alpha(infoMain, 0.8)} 100%)`,
          boxShadow: '0 8px 20px rgba(0, 0, 0, 0.15)',
        },
      },
    },
    MuiDrawer: {
      styleOverrides: {
        paper: {
          backgroundColor: '#FFFFFF',
          border: 'none',
          boxShadow: '0 12px 24px rgba(0, 0, 0, 0.12)',
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          boxShadow: '0 8px 16px rgba(0, 0, 0, 0.08)',
          borderRadius: 12,
          transition: 'box-shadow 0.3s, transform 0.3s',
          '&:hover': {
            boxShadow: '0 12px 24px rgba(0, 0, 0, 0.12)',
          },
        },
        elevation1: {
          boxShadow: '0 2px 12px rgba(0, 0, 0, 0.06)',
        },
        elevation2: {
          boxShadow: '0 4px 16px rgba(0, 0, 0, 0.08)',
        },
        elevation3: {
          boxShadow: '0 8px 20px rgba(0, 0, 0, 0.1)',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: 8,
          padding: '8px 16px',
          transition: 'all 0.2s ease-in-out',
          fontWeight: 500,
          '&:hover': {
            transform: 'translateY(-2px)',
          },
        },
        contained: {
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
          '&:hover': {
            boxShadow: '0 6px 16px rgba(0, 0, 0, 0.15)',
          },
        },
        containedPrimary: {
          background: `linear-gradient(45deg, ${primaryMain} 0%, ${alpha(infoMain, 0.8)} 100%)`,
          '&:hover': {
            background: `linear-gradient(45deg, ${primaryMain} 10%, ${alpha(infoMain, 0.9)} 90%)`,
          },
        },
        containedSecondary: {
          background: `linear-gradient(45deg, ${secondaryMain} 0%, ${alpha(successMain, 0.8)} 100%)`,
          '&:hover': {
            background: `linear-gradient(45deg, ${secondaryMain} 10%, ${alpha(successMain, 0.9)} 90%)`,
          },
        },
        outlinedPrimary: {
          borderWidth: 2,
          '&:hover': {
            borderWidth: 2,
          },
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 16,
          padding: 16,
          transition: 'transform 0.3s, box-shadow 0.3s',
          '&:hover': {
            transform: 'translateY(-5px)',
            boxShadow: '0 16px 32px rgba(0, 0, 0, 0.12)',
          },
        },
      },
    },
    MuiCardContent: {
      styleOverrides: {
        root: {
          padding: '24px 24px 24px',
          '&:last-child': {
            paddingBottom: 24,
          },
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: 8,
            transition: 'box-shadow 0.3s',
            '&:hover, &.Mui-focused': {
              boxShadow: `0 0 0 4px ${alpha(primaryMain, 0.1)}`,
            },
          },
        },
      },
    },
    MuiListItem: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          transition: 'all 0.2s',
          '&.Mui-selected': {
            backgroundColor: alpha(primaryMain, 0.08),
            '&:hover': {
              backgroundColor: alpha(primaryMain, 0.12),
            },
            '& .MuiListItemIcon-root': {
              color: primaryMain,
            },
          },
        },
        button: {
          '&:hover': {
            backgroundColor: alpha(primaryMain, 0.04),
            transform: 'translateX(4px)',
          },
        },
      },
    },
    MuiListItemIcon: {
      styleOverrides: {
        root: {
          minWidth: 40,
          color: '#6B7280',
        },
      },
    },
    MuiListItemText: {
      styleOverrides: {
        primary: {
          fontWeight: 500,
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: 8,
        },
        filled: {
          boxShadow: '0 2px 4px rgba(0, 0, 0, 0.075)',
        },
      },
    },
    MuiTooltip: {
      styleOverrides: {
        tooltip: {
          backgroundColor: neutralDark,
          borderRadius: 6,
          boxShadow: '0 4px 8px rgba(0, 0, 0, 0.15)',
          fontSize: 12,
        },
      },
    },
    MuiDataGrid: {
      styleOverrides: {
        root: {
          border: 'none',
          borderRadius: 12,
          '& .MuiDataGrid-columnHeaders': {
            backgroundColor: '#F9FAFB',
            borderTopLeftRadius: 12,
            borderTopRightRadius: 12,
          },
        },
        columnHeader: {
          paddingLeft: 16,
          paddingRight: 16,
        },
        cell: {
          paddingLeft: 16,
          paddingRight: 16,
        },
      },
    },
    MuiAlert: {
      styleOverrides: {
        root: {
          borderRadius: 10,
        },
        standard: {
          border: '1px solid',
        },
        standardSuccess: {
          backgroundColor: alpha(successMain, 0.12),
          borderColor: alpha(successMain, 0.2),
        },
        standardInfo: {
          backgroundColor: alpha(infoMain, 0.12),
          borderColor: alpha(infoMain, 0.2),
        },
        standardWarning: {
          backgroundColor: alpha(warningMain, 0.12),
          borderColor: alpha(warningMain, 0.2),
        },
        standardError: {
          backgroundColor: alpha(errorMain, 0.12),
          borderColor: alpha(errorMain, 0.2),
        },
      },
    },
    MuiAvatar: {
      styleOverrides: {
        root: {
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.15)',
        },
      },
    },
    MuiTab: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 500,
          borderTopLeftRadius: 10,
          borderTopRightRadius: 10,
          transition: 'all 0.2s',
          '&.Mui-selected': {
            backgroundColor: alpha(primaryMain, 0.08),
          },
        },
      },
    },
  },
});

export default theme; 
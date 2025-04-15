import React from 'react';
import {
  Container,
  Typography,
  Paper,
  Box,
  List,
  ListItem,
  ListItemIcon,
  ListItemText
} from '@mui/material';
import {
  TrendingUp, // For Z-score
  Forest,     // For Isolation Forest
  Grain,      // For DBSCAN
  AutoAwesomeMotion // For Autoencoder
} from '@mui/icons-material';

function ExplanationPage() {
  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" gutterBottom sx={{ fontWeight: 'bold', mb: 3, color: 'primary.main' }}>
        Методы Обнаружения Аномалий
      </Typography>

      <Paper sx={{ p: 3, mb: 3, borderRadius: 2, boxShadow: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <ListItemIcon sx={{ minWidth: 40, color: 'info.main' }}>
            <TrendingUp />
          </ListItemIcon>
          <Typography variant="h6" sx={{ fontWeight: 600 }}>Статистический метод (Z-Score)</Typography>
        </Box>
        <Typography variant="body1" paragraph>
          Этот метод основан на расчете Z-оценки (стандартного отклонения) для каждого числового признака данных.
          Z-оценка показывает, насколько далеко конкретное значение признака отстоит от среднего значения этого признака по выборке, измеренного в единицах стандартного отклонения.
        </Typography>
        <Typography variant="body1">
          <strong>Принцип работы:</strong>
          <List dense sx={{ pl: 2 }}>
            <ListItem>1. Для каждого числового признака (например, 'сумма заказа', 'количество действий за 5 минут') рассчитываются среднее значение (μ) и стандартное отклонение (σ) на обучающей ("нормальной") выборке.</ListItem>
            <ListItem>2. Для нового наблюдения и каждого признака рассчитывается Z-оценка: Z = (Значение - μ) / σ.</ListItem>
            <ListItem>3. Если абсолютное значение |Z| превышает заданный порог (например, 3), то наблюдение считается аномальным по этому признаку. Это означает, что значение признака находится более чем в 3 стандартных отклонениях от среднего, что маловероятно для нормального распределения.</ListItem>
          </List>
        </Typography>
        <Typography variant="subtitle2" sx={{ mt: 1, fontWeight: 'bold' }}>Настраиваемые параметры:</Typography>
        <List dense sx={{ pl: 2, listStyleType: 'disc', '& .MuiListItem-root': { display: 'list-item' } }}>
            <ListItem sx={{ py: 0.5 }}><strong>Порог Z-оценки (`z_threshold`):</strong> Определяет, насколько далеко от среднего (в стандартных отклонениях) должно быть значение, чтобы считаться аномалией. Увеличение порога делает детектор менее чувствительным.</ListItem>
        </List>
        <Typography variant="body2" color="text.secondary" paragraph sx={{ mt: 2 }}>
          <strong>Хорошо подходит для:</strong> Обнаружения выбросов в числовых данных, когда аномалии представляют собой значения, значительно отличающиеся от среднего (слишком большие или слишком маленькие). Прост в реализации и интерпретации.
        </Typography>
        <Typography variant="body2" color="text.secondary">
          <strong>Ограничения:</strong> Чувствителен к наличию выбросов в обучающей выборке (они могут исказить среднее и стандартное отклонение). Предполагает, что данные распределены примерно нормально (или, по крайней мере, симметрично).
        </Typography>
      </Paper>

      <Paper sx={{ p: 3, mb: 3, borderRadius: 2, boxShadow: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <ListItemIcon sx={{ minWidth: 40, color: 'success.main' }}>
            <Forest />
          </ListItemIcon>
          <Typography variant="h6" sx={{ fontWeight: 600 }}>Isolation Forest (Изолирующий Лес)</Typography>
        </Box>
        <Typography variant="body1" paragraph>
          Метод машинного обучения без учителя, основанный на построении ансамбля случайных "изолирующих" деревьев.
          Основная идея заключается в том, что аномальные точки данных легче "изолировать" (отделить от остальных), чем нормальные.
        </Typography>
        <Typography variant="body1">
          <strong>Принцип работы:</strong>
          <List dense sx={{ pl: 2 }}>
            <ListItem>1. Строится множество (лес) деревьев решений. Каждое дерево строится на случайной подвыборке данных.</ListItem>
            <ListItem>2. В каждом узле дерева случайно выбирается признак и случайное значение для разделения данных (между минимумом и максимумом признака в этом узле).</ListItem>
            <ListItem>3. Процесс повторяется, пока точка не будет изолирована в листе дерева.</ListItem>
            <ListItem>4. Аномальные точки, как правило, находятся дальше от основной массы данных, поэтому для их изоляции требуется меньше случайных разбиений (они оказываются на меньшей глубине в деревьях).</ListItem>
            <ListItem>5. Для каждой точки рассчитывается "оценка аномальности" на основе средней глубины ее изоляции по всем деревьям леса. Точки с наименьшей средней глубиной считаются наиболее аномальными.</ListItem>
          </List>
        </Typography>
        <Typography variant="body2" color="text.secondary" paragraph>
          <strong>Хорошо подходит для:</strong> Обнаружения различных типов выбросов в многомерных данных, особенно когда объем данных велик. Не требует предположений о распределении данных. Эффективен в пространствах высокой размерности.
        </Typography>
        <Typography variant="body2" color="text.secondary">
          <strong>Ограничения:</strong> Может быть менее эффективен, если аномалии образуют плотные кластеры. Может давать неоптимальные результаты на данных с большим количеством категориальных признаков (хотя OHE помогает).
        </Typography>
      </Paper>

      <Paper sx={{ p: 3, mb: 3, borderRadius: 2, boxShadow: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <ListItemIcon sx={{ minWidth: 40, color: 'warning.main' }}>
            <Grain />
          </ListItemIcon>
          <Typography variant="h6" sx={{ fontWeight: 600 }}>DBSCAN</Typography>
        </Box>
        <Typography variant="body1" paragraph>
          Алгоритм кластеризации, основанный на плотности. DBSCAN (Density-Based Spatial Clustering of Applications with Noise) группирует вместе точки, которые плотно расположены, отмечая как выбросы (шум) точки, которые лежат изолированно в областях с низкой плотностью.
        </Typography>
        <Typography variant="body1">
          <strong>Принцип работы:</strong>
          <List dense sx={{ pl: 2 }}>
            <ListItem>1. Алгоритм выбирает произвольную точку и находит все точки в ее окрестности заданного радиуса (ε, `eps`).</ListItem>
            <ListItem>2. Если в окрестности достаточное количество точек (больше или равно `min_samples`), эта точка считается "ядром", и начинается формирование кластера.</ListItem>
            <ListItem>3. Кластер расширяется путем добавления всех достижимых по плотности точек (точек в ε-окрестности ядерных точек).</ListItem>
            <ListItem>4. Точки, которые не являются ядерными и не достижимы из других ядерных точек, считаются шумом (аномалиями) и помечаются специальной меткой (обычно -1).</ListItem>
          </List>
        </Typography>
        <Typography variant="subtitle2" sx={{ mt: 1, fontWeight: 'bold' }}>Настраиваемые параметры:</Typography>
        <List dense sx={{ pl: 2, listStyleType: 'disc', '& .MuiListItem-root': { display: 'list-item' } }}>
            <ListItem sx={{ py: 0.5 }}><strong>Радиус окрестности (`dbscan_eps`):</strong> Максимальное расстояние между двумя точками, чтобы одна считалась соседом другой. Увеличение расширяет кластеры.</ListItem>
            <ListItem sx={{ py: 0.5 }}><strong>Минимальное число соседей (`dbscan_min_samples`):</strong> Количество точек, которое должно быть в окрестности точки (включая саму точку), чтобы она считалась ядром кластера. Увеличение делает детектор более строгим к шуму.</ListItem>
        </List>
        <Typography variant="body2" color="text.secondary" paragraph sx={{ mt: 2 }}>
          <strong>Хорошо подходит для:</strong> Обнаружения аномалий в виде "шума", который не формирует плотных скоплений. Способен находить кластеры произвольной формы, в отличие от методов вроде K-Means.
        </Typography>
        <Typography variant="body2" color="text.secondary">
          <strong>Ограничения:</strong> Требует подбора двух параметров (`eps` и `min_samples`), которые могут сильно влиять на результат. Плохо работает с данными, имеющими сильно различающуюся плотность кластеров.
        </Typography>
      </Paper>

      <Paper sx={{ p: 3, mb: 3, borderRadius: 2, boxShadow: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <ListItemIcon sx={{ minWidth: 40, color: 'secondary.main' }}>
            <AutoAwesomeMotion />
          </ListItemIcon>
          <Typography variant="h6" sx={{ fontWeight: 600 }}>Autoencoder (Автоэнкодер)</Typography>
        </Box>
        <Typography variant="body1" paragraph>
          Тип нейронной сети, используемый для неконтролируемого обучения признаков и снижения размерности. Автоэнкодер состоит из двух частей: энкодера, который сжимает входные данные в низкоразмерное "скрытое" представление (код), и декодера, который пытается восстановить исходные данные из этого кода.
        </Typography>
        <Typography variant="body1">
          <strong>Принцип работы:</strong>
          <List dense sx={{ pl: 2 }}>
            <ListItem>1. Сеть обучается на большом наборе "нормальных" данных (например, типичных пользовательских активностей).</ListItem>
            <ListItem>2. Цель обучения - минимизировать ошибку реконструкции, то есть разницу между исходными данными и данными, восстановленными декодером.</ListItem>
            <ListItem>3. После обучения, когда на вход подаются новые данные, автоэнкодер пытается их восстановить.</ListItem>
            <ListItem>4. Если входные данные являются "нормальными" (похожими на те, на которых обучалась сеть), ошибка реконструкции будет низкой.</ListItem>
            <ListItem>5. Если входные данные являются аномальными (сильно отличаются от обучающей выборки), сеть не сможет их хорошо восстановить, и ошибка реконструкции будет высокой.</ListItem>
            <ListItem>6. Наблюдения с ошибкой реконструкции выше определенного порога помечаются как аномалии.</ListItem>
          </List>
        </Typography>
        <Typography variant="subtitle2" sx={{ mt: 1, fontWeight: 'bold' }}>Настраиваемые параметры:</Typography>
        <List dense sx={{ pl: 2, listStyleType: 'disc', '& .MuiListItem-root': { display: 'list-item' } }}>
            <ListItem sx={{ py: 0.5 }}><strong>Использовать автоэнкодер (`use_autoencoder`):</strong> Включает или отключает использование этого детектора (применяется при запуске детекции для 'activity').</ListItem>
            <ListItem sx={{ py: 0.5 }}><strong>Порог ошибки реконструкции (`autoencoder_threshold`):</strong> Максимально допустимое значение ошибки реконструкции, при котором точка считается нормальной. Увеличение порога делает детектор менее чувствительным.</ListItem>
        </List>
        <Typography variant="body2" color="text.secondary" paragraph sx={{ mt: 2 }}>
          <strong>Хорошо подходит для:</strong> Обнаружения сложных, нелинейных аномалий, которые отклоняются от изученных "нормальных" паттернов данных. Может выявлять аномалии, структура которых заранее неизвестна. Полезен, когда "нормальное" поведение хорошо определено, а аномалии редки и разнообразны.
        </Typography>
        <Typography variant="body2" color="text.secondary">
          <strong>Ограничения:</strong> Требует достаточно большого набора "чистых" нормальных данных для обучения. Выбор архитектуры сети и параметров обучения может быть сложным. Может быть вычислительно более затратным, чем другие методы.
        </Typography>
      </Paper>

    </Container>
  );
}

export default ExplanationPage; 
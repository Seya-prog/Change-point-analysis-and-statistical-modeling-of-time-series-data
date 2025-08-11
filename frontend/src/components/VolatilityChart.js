import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Line, Bar } from 'react-chartjs-2';
import moment from 'moment';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const VolatilityChart = ({ data, volatility = [], returns = [] }) => {
  if (!data || !data.dates || !volatility.length) {
    return <div>No volatility data available</div>;
  }

  // Prepare labels
  const labels = data.dates.map(date => moment(date).format('MMM YY'));

  // Volatility chart options
  const volatilityOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Price Volatility Over Time (30-day rolling window)',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Volatility (USD)'
        }
      },
      x: {
        title: {
          display: true,
          text: 'Time Period'
        }
      }
    },
  };

  const volatilityData = {
    labels,
    datasets: [
      {
        label: 'Volatility',
        data: volatility,
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        fill: true,
      },
    ],
  };

  // Returns chart options
  const returnsOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Daily Returns Distribution',
      },
    },
    scales: {
      y: {
        title: {
          display: true,
          text: 'Daily Return (%)'
        }
      },
      x: {
        title: {
          display: true,
          text: 'Time Period'
        }
      }
    },
  };

  const returnsData = {
    labels,
    datasets: [
      {
        label: 'Daily Returns',
        data: returns,
        backgroundColor: returns.map(ret => ret >= 0 ? 'rgba(75, 192, 192, 0.6)' : 'rgba(255, 99, 132, 0.6)'),
        borderColor: returns.map(ret => ret >= 0 ? 'rgb(75, 192, 192)' : 'rgb(255, 99, 132)'),
        borderWidth: 1,
      },
    ],
  };

  return (
    <div>
      <div style={{ marginBottom: '40px' }}>
        <Line options={volatilityOptions} data={volatilityData} />
      </div>
      <div>
        <Bar options={returnsOptions} data={returnsData} />
      </div>
    </div>
  );
};

export default VolatilityChart;

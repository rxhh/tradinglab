import abc
import pandas as pd
from typing import Dict, Any
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import vectorbtpro as vbt
from . import indicators

class Signal(abc.ABC):
    @abc.abstractmethod
    def generate_signals(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate discrete trading signals based on input data and parameters.
        Must be implemented by concrete signal classes.
        
        df: DataFrame containing asset price data.
        params: Dictionary of parameters for signal generation.
        
        return: Dictionary containing backtesting data and signals.
                Keys: ['le', 'se', 'lx', 'sx'].
        """
        pass

    @abc.abstractmethod
    def optimize_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Optimize signal parameters based on historical data.
        Must be implemented by concrete signal classes.
        
        :param df: DataFrame containing asset price data.
        :return: Dictionary of optimized parameters.
        """
        pass

    def plot_signal_td_stop(self, df, params, max_td_stop=100):
        signals = self.generate_signals(df, params)
        pf = vbt.Portfolio.from_signals(
            df['close'], open=df['open'], high=df['high'], low=df['low'],
            entries=signals['le'], short_entries=signals['se'],
            td_stop=vbt.Param(np.arange(2,max_td_stop)),
            time_delta_format=0,
        )
        fig, ax = plt.subplots(2, 2, sharex=True, figsize=(12, 8))
        pf.total_return.plot(title='Total Return', ax=ax[0,0], marker='.')
        pf.trades.win_rate.plot(title='Win Rate', ax=ax[1,0], marker='.')
        pf.trades.expectancy.plot(title='Expectancy', ax=ax[0,1], marker='.')
        pf.max_drawdown.plot(title='Max Drawdown', ax=ax[1,1], marker='.')
        fig.suptitle("Stats by Time Exit")
        fig.tight_layout()
        return
    
    def plot_signal_sltp_stop_natr(self, df, params, atr_len=20, sls=np.arange(1, 10, 0.5), tps=np.arange(1, 10, 0.5)):
        signals = self.generate_signals(df, params)
        natr = indicators.SuperSmoother(df['tr'], atr_len)/df['close']
        sl_stops = []
        tp_stops = []
        for sl in sls:
            for tp in tps:            
                sl_stops.append((natr*sl).rename((sl, tp)))
                tp_stops.append((natr*tp).rename((sl, tp)))
        sl_stops = pd.concat(sl_stops, axis=1)
        sl_stops.columns = sl_stops.columns.set_names(['sl', 'tp'])
        tp_stops = pd.concat(tp_stops, axis=1)
        tp_stops.columns = tp_stops.columns.set_names(['sl', 'tp'])
        pf = vbt.Portfolio.from_signals(
            df['close'], open=df['open'], high=df['high'], low=df['low'],
            entries=signals['le'], short_entries=signals['se'],
            sl_stop=sl_stops,
            tp_stop=tp_stops,
        )
        fig, ax = plt.subplots(2, 2, sharex=True, figsize=(12, 8))
        sns.heatmap(self._round_index(pf.total_return.unstack()), ax=ax[0,0]).set_title('Total Return')
        sns.heatmap(self._round_index(pf.trades.win_rate.unstack()), ax=ax[1,0]).set_title('Win Rate')
        sns.heatmap(self._round_index(pf.trades.expectancy.unstack()), ax=ax[0,1]).set_title('Expectancy')
        sns.heatmap(self._round_index(pf.max_drawdown.unstack()), ax=ax[1,1]).set_title('Max Drawdown')
        fig.suptitle(f"Stats by SL/TP in NATR({atr_len}) Multiples")
        fig.tight_layout()
        return

    def plot_signal_sltp_stop_pct(self, df, params, sls=np.arange(0.005, 0.1, 0.005), tps=np.arange(0.005, 0.1, 0.005)):
        signals = self.generate_signals(df, params)
        pf = vbt.Portfolio.from_signals(
            df['close'], open=df['open'], high=df['high'], low=df['low'],
            entries=signals['le'], short_entries=signals['se'],
            sl_stop=vbt.Param(sls),
            tp_stop=vbt.Param(tps),
        )
        fig, ax = plt.subplots(2, 2, sharex=True, figsize=(12, 8))
        sns.heatmap(self._round_index(pf.total_return.unstack()), ax=ax[0,0]).set_title('Total Return')
        sns.heatmap(self._round_index(pf.trades.win_rate.unstack()), ax=ax[1,0]).set_title('Win Rate')
        sns.heatmap(self._round_index(pf.trades.expectancy.unstack()), ax=ax[0,1]).set_title('Expectancy')
        sns.heatmap(self._round_index(pf.max_drawdown.unstack()), ax=ax[1,1]).set_title('Max Drawdown')
        fig.suptitle(f"Stats by SL/TP in Pct")
        fig.tight_layout()
        return
    
    def _round_index(self, d, n=3):
        d.index = d.index.round(n)
        d.columns = d.columns.round(n)
        return d
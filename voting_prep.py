"""
🗳 VOTING PREP MODULE
Focused workflow to generate next-epoch volume predictions per pool
and suggest vote weights for the upcoming Full Sail Finance epoch.
"""

import io
import json
from typing import Dict, List
import numpy as np
import pandas as pd
import streamlit as st

from epoch_volume_predictor import EpochVolumePredictor


@st.cache_data(show_spinner=False)
def _cached_epoch_prediction(pool_name: str, df_json: str) -> Dict:
    """Cache wrapper for epoch prediction keyed by serialized data."""
    evp = EpochVolumePredictor()
    df = pd.read_json(io.StringIO(df_json))
    return evp.predict_next_epoch_volume(df, pool_name)


class VotingPrep:
    """Render Voting Prep workflow for upcoming epoch voting."""

    def __init__(self):
        self.epoch_predictor = EpochVolumePredictor()
        # Storage path for weekly predictions
        self._predictions_dir = 'data_cache/weekly_predictions'

    def _ensure_predictions_dir(self):
        import os
        if not os.path.exists(self._predictions_dir):
            os.makedirs(self._predictions_dir, exist_ok=True)

    def _epoch_file_path(self, epoch_id: str) -> str:
        import os
        self._ensure_predictions_dir()
        return os.path.join(self._predictions_dir, f"epoch_{epoch_id}.json")

    def _load_saved_epoch_predictions(self, epoch_id: str) -> List[Dict]:
        import os
        path = self._epoch_file_path(epoch_id)
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save_epoch_predictions(self, epoch_id: str, predictions: List[Dict]):
        path = self._epoch_file_path(epoch_id)
        try:
            with open(path, 'w') as f:
                json.dump(predictions, f)
        except Exception:
            pass

    def _ensure_processed_pools(self) -> Dict[str, pd.DataFrame]:
        """Ensure we have per-pool processed data in session state.

        Returns a mapping from pool name to DataFrame with at least
        columns ['date','pool','volume_24h'].
        """
        if 'processed_data' in st.session_state and st.session_state.processed_data:
            return st.session_state.processed_data

        # Fallback: if only a single historical dataframe exists, split by pool if present
        if 'historical_data' in st.session_state and not st.session_state.historical_data.empty:
            df: pd.DataFrame = st.session_state.historical_data.copy()
            if 'pool' in df.columns:
                pools: Dict[str, pd.DataFrame] = {
                    pool: grp.sort_values('date') for pool, grp in df.groupby('pool')
                }
                return pools
            # If no pool column, treat as a generic single-pool dataset
            return {'SAIL/USDC': df.sort_values('date')}

        return {}

    def _predict_epoch_for_pool(self, pool_name: str, pool_df: pd.DataFrame) -> Dict:
        try:
            # Keep minimal required columns
            cols = [c for c in ['date', 'volume_24h'] if c in pool_df.columns]
            df = pool_df[cols].dropna()
            df = df.sort_values('date')
            # Serialize for cache key (compact)
            df_ser = df.to_json(orient='records', date_format='iso')
            return _cached_epoch_prediction(pool_name, df_ser)
        except Exception as e:
            return {'pool': pool_name, 'error': str(e)}

    def _top_pools_by_recent_volume(self, pools: Dict[str, pd.DataFrame], days: int = 14, top_k: int = 6) -> List[str]:
        """Rank pools by recent volume and return top_k names."""
        scores = []
        cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=days)
        for name, df in pools.items():
            if 'date' not in df.columns or 'volume_24h' not in df.columns:
                continue
            sdf = df.copy()
            try:
                sdf['date'] = pd.to_datetime(sdf['date']).dt.normalize()
            except Exception:
                pass
            # Ensure cutoff is also normalized for comparison
            recent = sdf[sdf['date'] >= cutoff.normalize()]
            if not recent.empty:
                vol = float(recent['volume_24h'].clip(lower=0).sum())
            else:
                vol = float(sdf['volume_24h'].clip(lower=0).tail(days).sum()) if 'volume_24h' in sdf else 0.0
            scores.append((name, vol))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [n for n, _ in scores[:top_k]] if scores else list(pools.keys())[:top_k]

    def _suggest_vote_weights(self, predictions: List[Dict], method: str = 'volume_share') -> pd.DataFrame:
        rows = []
        for p in predictions:
            if 'error' in p:
                continue
            rows.append({
                'pool': p['pool'],
                'predicted_epoch_volume': float(p['predicted_epoch_volume']),
                'lower_bound': float(p['lower_bound']),
                'upper_bound': float(p['upper_bound']),
                'confidence_score': float(p.get('confidence_score', 0.0)),
                'model_accuracy': float(p.get('model_accuracy', 0.0)),
                'prediction_range_pct': float(p.get('prediction_range_pct', 0.0))
            })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        if method == 'volume_share':
            total = df['predicted_epoch_volume'].sum()
            df['vote_weight'] = np.where(total > 0, df['predicted_epoch_volume'] / total, 0.0)
        elif method == 'confidence_adjusted':
            score = df['predicted_epoch_volume'] * (0.5 + 0.5 * df['confidence_score'])
            total = score.sum()
            df['vote_weight'] = np.where(total > 0, score / total, 0.0)
        elif method == 'risk_adjusted':
            # Penalize wide ranges
            score = df['predicted_epoch_volume'] * (1.0 - df['prediction_range_pct'].clip(0, 0.9))
            total = score.sum()
            df['vote_weight'] = np.where(total > 0, score / total, 0.0)
        else:
            total = df['predicted_epoch_volume'].sum()
            df['vote_weight'] = np.where(total > 0, df['predicted_epoch_volume'] / total, 0.0)

        df = df.sort_values('vote_weight', ascending=False)
        return df

    def _download_csv_button(self, df: pd.DataFrame, filename: str):
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        st.download_button(
            label='⬇️ Download CSV',
            data=csv_buf.getvalue(),
            file_name=filename,
            mime='text/csv'
        )

    def render(self):
        st.subheader("🗳 Voting Prep - Next Epoch Forecasts")

        # Determine epoch context and display next-epoch window
        try:
            epoch_info = self.epoch_predictor.epoch_predictor.get_current_epoch_info()  # nested within EpochVolumePredictor
        except Exception:
            epoch_info = None

        next_epoch_number = None
        window_text = None
        next_epoch_start = None
        if epoch_info and isinstance(epoch_info, dict):
            next_epoch_number = epoch_info.get('epoch_number', None)
            # Try to build a date window string if available
            start = epoch_info.get('next_epoch_start') or epoch_info.get('epoch_end')
            end = epoch_info.get('next_epoch_end')
            if start and end:
                window_text = f"{pd.to_datetime(start).date()} → {pd.to_datetime(end).date()}"
            if start:
                next_epoch_start = pd.to_datetime(start)

        if window_text is None:
            # Fallback to 7-day window from now (UTC)
            start_fallback = pd.Timestamp.utcnow().normalize()
            end_fallback = start_fallback + pd.Timedelta(days=7)
            window_text = f"{start_fallback.date()} → {end_fallback.date()}"

        st.info(f"All predictions below are estimates for the NEXT 7-DAY EPOCH window: {window_text}")

        pools = self._ensure_processed_pools()
        if not pools:
            st.warning("No processed pool data found. Load data first from the sidebar.")
            return

        # ---- Auto-generate scheduled weekly predictions ----
        epoch_id = str(next_epoch_number) if next_epoch_number is not None else window_text or 'unknown'
        saved_preds = self._load_saved_epoch_predictions(epoch_id)
        now_utc = pd.Timestamp.utcnow()
        about_to_start = False
        if next_epoch_start is not None:
            # user wants starting in about 2 hours
            about_to_start = now_utc >= (next_epoch_start - pd.Timedelta(hours=2))

        cstat1, cstat2, cstat3 = st.columns(3)
        with cstat1:
            st.caption(f"Epoch ID: {epoch_id}")
        with cstat2:
            if next_epoch_start is not None:
                starts = next_epoch_start.tz_localize('UTC') if next_epoch_start.tzinfo is None else next_epoch_start
                st.caption(f"Next epoch starts: {starts}")
        with cstat3:
            st.caption(f"Saved predictions: {'Yes' if saved_preds else 'No'}")

        run_auto = False
        colx, coly = st.columns([1,1])
        with colx:
            if st.button("⚙️ Run Weekly Predictions Now", help="Generate predictions for all pools for the upcoming epoch and save"):
                run_auto = True
        with coly:
            st.toggle("Auto-run ~2h before epoch", key='auto_weekly_toggle', value=True, help="Runs automatically when the app is open ~2 hours before the epoch starts")
        if about_to_start and not saved_preds:
            st.warning("Epoch starts in ~2 hours. Auto-run is enabled; predictions will generate automatically if the app is open.")

        should_auto_run = st.session_state.get('auto_weekly_toggle', True) and about_to_start and not saved_preds
        if run_auto or should_auto_run:
            with st.spinner("Generating weekly predictions for all pools..."):
                all_preds: List[Dict] = []
                for pool_name, pool_df in pools.items():
                    pred = self._predict_epoch_for_pool(pool_name, pool_df)
                    pred['epoch_id'] = epoch_id
                    all_preds.append(pred)
                self._save_epoch_predictions(epoch_id, all_preds)
                st.session_state['weekly_predictions'] = {epoch_id: all_preds}
                st.success("Weekly predictions generated and saved.")
                saved_preds = all_preds

        # Pool selection
        pool_list = list(pools.keys())
        default_selection = self._top_pools_by_recent_volume(pools, days=14, top_k=min(6, len(pool_list)))
        selected_pools = st.multiselect(
            "Select pools to include",
            options=pool_list,
            default=default_selection
        )

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            method = st.selectbox(
                "Vote weight method",
                ["volume_share", "confidence_adjusted", "risk_adjusted"],
                index=1,
                help="How to convert predictions into suggested vote weights"
            )
        with col_b:
            show_daily = st.checkbox("Show daily breakdown", value=False)
        with col_c:
            normalize_100 = st.checkbox("Normalize to 100%", value=True)

        # Generate predictions (use saved epoch predictions if available)
        predictions: List[Dict] = []
        if saved_preds:
            # Filter saved predictions to selected pools
            sel = {p for p in selected_pools}
            predictions = [p for p in saved_preds if p.get('pool') in sel]
            # If some selected pools missing, backfill
            missing = [p for p in selected_pools if p not in {x.get('pool') for x in predictions}]
            if missing:
                with st.spinner("Generating missing forecasts..."):
                    for pool in missing:
                        predictions.append(self._predict_epoch_for_pool(pool, pools[pool]))
        else:
            with st.spinner("Generating next-epoch forecasts..."):
                for pool in selected_pools:
                    pred = self._predict_epoch_for_pool(pool, pools[pool])
                    predictions.append(pred)

        # Display per-pool cards
        cards_per_row = 3
        for i in range(0, len(predictions), cards_per_row):
            cols = st.columns(cards_per_row)
            for j, pred in enumerate(predictions[i:i + cards_per_row]):
                with cols[j]:
                    if 'error' in pred:
                        st.error(f"{pred.get('pool','?')}: {pred['error']}")
                        continue
                    st.markdown(f"**{pred['pool']}**")
                    st.metric("Predicted Epoch Volume", f"${pred['predicted_epoch_volume']:,.0f}")
                    st.caption(f"Range: ${pred['lower_bound']:,.0f} - ${pred['upper_bound']:,.0f}")
                    st.caption(f"Confidence: {pred.get('confidence_score',0.0):.0%} • Accuracy: {pred.get('model_accuracy',0.0):.0%}")
                    if show_daily:
                        daily = pred.get('daily_predictions')
                        if isinstance(daily, pd.DataFrame):
                            st.dataframe(daily, use_container_width=True, height=240)

        # Suggest vote weights
        weights_df = self._suggest_vote_weights(predictions, method=method)
        if weights_df.empty:
            st.warning("No valid predictions to compute vote weights.")
            return

        if normalize_100:
            weights_df['vote_weight_pct'] = (weights_df['vote_weight'] * 100).round(2)
        else:
            weights_df['vote_weight_pct'] = weights_df['vote_weight']

        st.markdown("### ✅ Suggested Vote Weights")
        st.dataframe(
            weights_df[[
                'pool', 'predicted_epoch_volume', 'vote_weight_pct', 'confidence_score', 'model_accuracy'
            ]].rename(columns={'vote_weight_pct': 'vote_weight (%)'}),
            use_container_width=True
        )

        self._download_csv_button(weights_df, filename='voting_prep_weights.csv')

        # Recommended vote slate (clear, ready to submit)
        st.markdown("### 📝 Recommended Vote Slate (This Epoch)")
        slate_df = weights_df[['pool', 'vote_weight']].copy()
        # Normalize precisely to 100%
        total = slate_df['vote_weight'].sum()
        if total > 0:
            slate_df['pct'] = (slate_df['vote_weight'] / total * 100)
        else:
            slate_df['pct'] = 0.0
        # Round and re-normalize last entry to ensure sum=100.00
        slate_df = slate_df.sort_values('pct', ascending=False).reset_index(drop=True)
        slate_df['pct'] = slate_df['pct'].round(2)
        diff = 100.0 - slate_df['pct'].sum()
        if len(slate_df) > 0:
            slate_df.loc[0, 'pct'] = (slate_df.loc[0, 'pct'] + diff).round(2)

        # Display as ranked list
        for idx, row in slate_df.iterrows():
            st.markdown(f"{idx+1}. **{row['pool']}** — **{row['pct']:.2f}%**")

        # Copyable text for submission
        slate_lines = [f"{row['pool']},{row['pct']:.2f}%" for _, row in slate_df.iterrows()]
        slate_text = "\n".join(slate_lines)
        st.text_area("Copy & paste for vote submission", value=slate_text, height=120)

        # Save/restore slate controls
        st.markdown("### 💾 Save or Restore Slate")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Save Slate to Session"):
                st.session_state.saved_voting_slate = slate_df[['pool', 'pct']].to_dict(orient='records')
                st.success("Slate saved for this session.")
        with c2:
            slate_json = json.dumps(slate_df[['pool', 'pct']].to_dict(orient='records'), indent=2)
            st.download_button("Download Slate JSON", data=slate_json, file_name="voting_slate.json", mime="application/json")
        with c3:
            up = st.file_uploader("Upload Slate JSON", type=["json"], label_visibility="collapsed")
            if up is not None:
                try:
                    data = json.load(up)
                    if isinstance(data, list) and all('pool' in d and 'pct' in d for d in data):
                        st.session_state.saved_voting_slate = data
                        st.success("Slate loaded into session.")
                    else:
                        st.warning("Invalid slate format.")
                except Exception as e:
                    st.error(f"Failed to load slate: {e}")

        # Explanations & sanity checks
        st.markdown("### 🧭 Weighting Method Explanation")
        expl = {
            'volume_share': "Weights proportional to predicted epoch volume.",
            'confidence_adjusted': "Volume share boosted by model confidence (50-100%).",
            'risk_adjusted': "Volume share penalized when prediction range is wide (risk-aware)."
        }
        st.caption(expl.get(method, expl['volume_share']))

        issues = []
        # Flag low-confidence pools or very wide ranges
        low_conf = weights_df[weights_df['confidence_score'] < 0.5]
        if not low_conf.empty:
            issues.append(f"{len(low_conf)} low-confidence pools (<50%).")
        wide = weights_df[weights_df['prediction_range_pct'] > 0.5]
        if not wide.empty:
            issues.append(f"{len(wide)} pools have very wide prediction ranges (>50%).")
        if issues:
            st.warning("; ".join(issues))



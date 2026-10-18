"""Run with: uv run streamlit run streamlit_app.py"""

from datetime import UTC, date, datetime, timedelta

import pandas as pd
import streamlit as st
from botocore.exceptions import BotoCoreError, ClientError

from trial_conversion_model.dashboard_data import (
    choose_verdicts,
    connect,
    list_artifacts,
    parse_scores,
    parse_verdict,
    read_bytes,
)

st.set_page_config(page_title='Beam | Trial conversion', page_icon='📊', layout='wide')


@st.cache_data(ttl=300, show_spinner=False)
def inventory(bucket, _client):
    return (
        list_artifacts(_client, bucket, 'predictions/', '.csv'),
        list_artifacts(_client, bucket, 'monitoring/', '.json'),
    )


@st.cache_data(ttl=300, show_spinner=False, max_entries=256)
def scores(bucket, artifact, _client):
    return parse_scores(read_bytes(_client, bucket, artifact))


@st.cache_data(ttl=300, show_spinner=False, max_entries=256)
def verdict(bucket, artifact, _client):
    return parse_verdict(read_bytes(_client, bucket, artifact))


def main():
    st.caption('BEAM / GROWTH INTELLIGENCE')
    st.title('Trial conversion monitor')
    st.write('Prioritize today’s outreach. Track how your trial population is changing.')
    with st.sidebar:
        st.header('Dashboard controls')
        if st.button('Refresh from S3', type='primary', width='stretch'):
            st.cache_data.clear()
        st.caption('S3 reads are cached for 5 minutes. Refresh checks for new artifacts immediately.')
    try:
        bucket, client = connect()
        with st.spinner('Reading S3 artifacts…'):
            predictions, reports = inventory(bucket, client)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    except (BotoCoreError, ClientError):
        st.error('Cannot read S3. Check AWS credentials, region, and s3:ListBucket / s3:GetObject permissions.')
        st.stop()
    st.sidebar.caption(f'Source: s3://{bucket}')
    if not predictions and not reports:
        st.info('No dated prediction CSVs or monitoring JSON reports were found.')
        return

    end = max(a.day for a in predictions + reports)
    months = sorted({a.day.strftime('%Y-%m') for a in predictions + reports}, reverse=True)
    period = st.sidebar.selectbox('History window', ['Latest 31 days', 'Calendar month'])
    if period == 'Calendar month':
        month = st.sidebar.selectbox('Month', months)
        start = date.fromisoformat(month + '-01')
        stop = (start.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
    else:
        start, stop = end - timedelta(days=30), end
    cutoff = st.sidebar.slider('Outreach: conversion probability below', 0.0, 1.0, 0.5, 0.05, format='%.2f')
    st.sidebar.caption('Rank 1 is the trial least likely to convert. The cutoff is an outreach filter, not a drift threshold.')

    records, failures = [], []
    latest_report_day = max((a.day for a in reports), default=None)
    for artifact in reports:
        if start <= artifact.day <= stop or artifact.day == latest_report_day:
            try:
                records.append((artifact, verdict(bucket, artifact, client)))
            except (ValueError, BotoCoreError, ClientError):
                failures.append(artifact.key)
    selected = choose_verdicts(records)
    latest_verdict = selected.get(latest_report_day)
    latest = predictions[-1] if predictions else None
    frame = None
    if latest:
        try:
            frame = scores(bucket, latest, client)
        except (ValueError, BotoCoreError, ClientError):
            failures.append(latest.key)

    st.subheader('Latest snapshot')
    a, b, c, d = st.columns(4)
    a.metric('Scored trials', f'{len(frame):,}' if frame is not None else 'Unavailable')
    b.metric('Mean predicted conversion', f"{frame.conversion_probability.mean():.1%}" if frame is not None and len(frame) else '—')
    c.metric('Outreach candidates', str(int((frame.conversion_probability < cutoff).sum())) if frame is not None else '—')
    d.metric('Features drifted', f"{latest_verdict[1]['drifted_share']:.0%}" if latest_verdict else 'Unknown')
    if latest:
        st.caption(f'Latest scoring date: {latest.day} · {latest.key}')
        if latest.day < datetime.now(UTC).date():
            st.warning(f'Latest predictions are {(datetime.now(UTC).date() - latest.day).days} day(s) old (UTC).')
    if latest_verdict:
        artifact, report = latest_verdict
        label = 'DRIFT DETECTED' if report['drift'] else 'NO DRIFT DETECTED'
        (st.error if report['drift'] else st.success)(f"{label} · {artifact.day} · {report['drifted_share']:.0%} of features drifted")
        st.caption(f"Monitoring source: {artifact.key} · {'Synthetic backfill' if report.get('source') == 'backfill' else 'Measured report'}")
        if latest and artifact.day != latest.day:
            st.warning('The latest drift report and predictions cover different dates.')
    else:
        st.warning('Latest drift verdict unavailable. Missing or invalid reports do not imply no drift.')

    ranked, history = st.tabs(['Latest ranked list', 'Month of history'])
    with ranked:
        if frame is None:
            st.info('Latest predictions are unavailable.')
        else:
            st.caption('Lowest conversion probability first; highest cancellation risk first. Scores are predictions, not observed conversions.')
            view = frame.copy()
            filters = st.columns(3)
            for col, field in zip(filters[:2], ['country', 'device_type']):
                if field in view:
                    choices = col.multiselect(field.replace('_', ' ').title(), sorted(frame[field].dropna().unique()))
                    if choices:
                        view = view[view[field].isin(choices)]
            if filters[2].checkbox('Outreach candidates only'):
                view = view[view.conversion_probability < cutoff]
            st.caption(f'{len(view):,} of {len(frame):,} trials · original cohort ranks preserved')
            st.dataframe(view, hide_index=True, width='stretch', column_config={
                'conversion_probability': st.column_config.ProgressColumn('Conversion probability', min_value=0, max_value=1, format='%.3f'),
                'cancellation_risk': st.column_config.ProgressColumn('Cancellation risk', min_value=0, max_value=1, format='%.3f'),
            })
            st.download_button('Download filtered ranked list', view.to_csv(index=False).encode(), f'ranked_{latest.day}.csv', 'text/csv')

    with history:
        st.caption(f'{start} through {stop} · dates come from filenames, not S3 upload times')
        daily = {a.day: a for a in predictions if start <= a.day <= stop}
        rows = []
        with st.spinner('Loading daily history…'):
            for timestamp in pd.date_range(start, stop):
                day = timestamp.date()
                row = {'date': timestamp, 'scored_trials': None, 'mean_conversion': None, 'outreach_candidates': None,
                       'drifted_share': None, 'verdict': 'Missing', 'source': 'Missing'}
                if day in daily:
                    try:
                        data = scores(bucket, daily[day], client)
                        row.update(scored_trials=len(data), mean_conversion=data.conversion_probability.mean(),
                                   outreach_candidates=int((data.conversion_probability < cutoff).sum()))
                    except (ValueError, BotoCoreError, ClientError):
                        failures.append(daily[day].key)
                if day in selected:
                    _, report = selected[day]
                    row.update(drifted_share=report['drifted_share'], verdict='Drift' if report['drift'] else 'No drift',
                               source='Synthetic backfill' if report.get('source') == 'backfill' else 'Measured report')
                rows.append(row)
        table = pd.DataFrame(rows)
        x, y = st.columns(2)
        x.metric('Days with predictions', f"{table.scored_trials.notna().sum()} / {len(table)}")
        y.metric('Days with drift detected', int(table.verdict.eq('Drift').sum()))
        st.info('Measured drift reports take precedence over synthetic backfill on the same date. Backfilled verdicts are illustrative history, not measured checks. Missing days remain gaps.')
        x, y = st.columns(2)
        with x:
            st.markdown('**Predicted conversion over time**')
            st.line_chart(table.set_index('date')[['mean_conversion']], color='#2563eb')
            st.markdown('**Daily scoring volume**')
            st.bar_chart(table.set_index('date')[['scored_trials']], color='#14b8a6')
        with y:
            st.markdown('**Share of features drifted**')
            measured = table.drifted_share.where(table.source.eq('Measured report'))
            backfill = table.drifted_share.where(table.source.eq('Synthetic backfill'))
            chart = pd.DataFrame({'date': table.date, 'Measured': measured, 'Backfill': backfill}).set_index('date')
            st.line_chart(chart, color=['#ef4444', '#94a3b8'])
            st.caption('Shares range from 0 to 1. The stored boolean verdict is authoritative.')
        st.dataframe(table, hide_index=True, width='stretch')
        st.download_button('Download daily history', table.to_csv(index=False).encode(), f'history_{start}_{stop}.csv', 'text/csv')
    if failures:
        st.warning('Some artifacts could not be read or validated; affected results are unavailable.')
        with st.expander('Affected S3 keys'):
            st.write(sorted(set(failures)))


if __name__ == '__main__':
    main()

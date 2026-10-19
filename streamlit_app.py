"""Run with: uv run streamlit run streamlit_app.py"""

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

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

st.set_page_config(page_title='Beam | Trial conversion', page_icon=':material/query_stats:', layout='wide')


@st.cache_data(ttl=300, show_spinner=False, max_entries=16)
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
    graphic, introduction = st.columns([1, 3], gap='large', vertical_alignment='top')
    with graphic:
        st.image(
            str(Path(__file__).resolve().parent / 'assets' / 'beam-trial-conversion-infographic.png'),
            width=650,
            alt='Beam trial conversion workflow: observe early engagement, predict conversion, prioritize outreach, and monitor drift.',
        )
    with introduction:
        st.caption('BEAM / MACHINE LEARNING PORTFOLIO DEMO')
        st.title('Trial conversion monitor')
        st.write(
            'This portfolio project tackles early trial conversion prediction for Beam, a subscription '
            'media app used in a simulated B2B SaaS training scenario. In the business case, roughly '
            '250 users begin a 14-day free trial each week, but only about half convert to a paid plan, '
            'below the 58% planning assumption. Trials convert automatically unless users cancel. '
            'Instead of waiting until a trial ends, the growth team needs to identify users who may '
            'cancel while there is still time to help. The model uses the first three days of activity '
            'to predict paid conversion by day 14; this dashboard turns those scores into a ranked '
            'outreach list and tracks changes in the trial population.'
        )
        with st.container(border=True):
            st.markdown('**The data**')
            st.write(
                'This demo uses synthetic trial and engagement data, not real customer records. '
                'Each row in the ranked list represents one user’s trial. The model uses the first '
                'three days of activity—session counts, listening sessions, time spent, country, '
                'and device type—to estimate whether a 14-day free trial will convert to a paid subscription. '
                'Historical training records include the outcome: converted (1) or did not convert (0). '
                'Training data and daily scoring cohorts come from the same synthetic generator, '
                'designed to mimic realistic trial behavior. The course deliberately shifts later '
                'signups toward lower-intent users to demonstrate data drift.'
            )
            st.markdown('**The purpose**')
            st.write(
                'Scoring on day 3 leaves 11 days for a growth team to offer onboarding support. '
                'The Outreach view puts trials least likely to convert first; use the outreach '
                'cutoff and country/device filters to explore potential candidates. The History view '
                'shows scoring volume, predicted conversion, and data drift—changes in the input '
                'population compared with reference data.'
            )
            st.caption(
                'An ML/MLOps demonstration: data preparation, model training, batch scoring, and monitoring. '
                'This dashboard displays saved results; it does not train models or run drift checks. '
                'Probabilities are predictions, not observed conversion rates or proof that outreach works. '
                'History labeled Synthetic backfill contains illustrative drift verdicts, not measured checks.'
            )
    try:
        bucket, client = connect()
        with st.spinner('Loading demo results…'):
            predictions, reports = inventory(bucket, client)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    except (BotoCoreError, ClientError):
        st.error('Cannot read S3. Check AWS credentials, region, and s3:ListBucket / s3:GetObject permissions.')
        st.stop()
    if not predictions and not reports:
        st.info('No dated prediction CSVs or monitoring JSON reports were found.')
        return

    end = max(a.day for a in predictions + reports)
    months = sorted({a.day.strftime('%Y-%m') for a in predictions + reports}, reverse=True)
    page = st.segmented_control('View', ['Outreach', 'History'], default='Outreach', key='view') or 'Outreach'
    start, stop = end - timedelta(days=30), end
    if page == 'History':
        with st.container(horizontal=True, vertical_alignment='bottom'):
            period = st.selectbox('History window', ['Latest 31 days', 'Calendar month'], width=240)
            if period == 'Calendar month':
                month = st.selectbox('Month', months, width=200)
                start = date.fromisoformat(month + '-01')
                stop = (start.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
    with st.expander('How to read this dashboard'):
        st.write('Scores estimate conversion; they are not observed outcomes. Rank 1 is the trial '
                 'with the lowest predicted conversion probability. Use the list to review outreach '
                 'opportunities, not as evidence that outreach will change an outcome.')
        st.write('Drift describes changes in input features. It does not measure prediction accuracy. '
                 'Synthetic backfill is illustrative and is not a measured monitoring check.')
    cutoff = st.slider('Outreach cutoff · predicted conversion below', 0, 100, 50, 5,
                       format='%d%%', key='cutoff',
                       help='Strictly below this percentage. This filters outreach candidates; it does not change model scores or drift verdicts.') / 100

    records, failures = [], []
    latest_report_day = max((a.day for a in reports), default=None)
    for artifact in reports:
        if (page == 'History' and start <= artifact.day <= stop) or artifact.day == latest_report_day:
            try:
                records.append((artifact, verdict(bucket, artifact, client)))
            except (ValueError, UnicodeError, pd.errors.ParserError, BotoCoreError, ClientError):
                failures.append(artifact.key)
    selected = choose_verdicts(records)
    latest_verdict = selected.get(latest_report_day)
    latest = predictions[-1] if predictions else None
    frame = None
    if latest:
        try:
            frame = scores(bucket, latest, client)
        except (ValueError, UnicodeError, pd.errors.ParserError, BotoCoreError, ClientError):
            failures.append(latest.key)

    if page == 'Outreach':
        st.subheader('Latest scored cohort')
        if latest:
            st.caption(f'Scored {latest.day:%b %d, %Y} · summary covers the full cohort')
        with st.container(horizontal=True):
            st.metric('Scored trials', f'{len(frame):,}' if frame is not None else '—', border=True, width=230)
            st.metric('Mean conversion score', f"{frame.conversion_probability.mean():.1%}" if frame is not None and len(frame) else '—', border=True, width=230)
            st.metric(f'Below {cutoff:.0%} cutoff', f'{int((frame.conversion_probability < cutoff).sum()):,}' if frame is not None else '—', border=True, width=230)
        if latest and latest.day < datetime.now(UTC).date():
            st.warning(f'Predictions are {(datetime.now(UTC).date() - latest.day).days} day(s) old (UTC). Review freshness before using this list.')
        if latest_verdict:
            artifact, report = latest_verdict
            label = 'Drift detected' if report['drift'] else 'No drift detected'
            detail = f"{label} · {artifact.day} · {report['drifted_share']:.0%} of features drifted."
            if report.get('source') == 'backfill':
                st.info(f'Synthetic backfill · {detail} This is illustrative, not a measured check.')
            else:
                (st.warning if report['drift'] else st.success)(f'Measured monitoring · {detail}')
            if latest and artifact.day != latest.day:
                st.caption('Monitoring and predictions cover different dates.')
        else:
            st.warning('Latest drift verdict unavailable. Missing or invalid reports do not imply no drift.')
        with st.expander('Data sources'):
            if latest:
                st.code(latest.key, language=None)
            if latest_verdict:
                st.code(latest_verdict[0].key, language=None)

    if page == 'Outreach':
        if frame is None:
            st.info('Latest predictions are unavailable. Please try again later.')
        else:
            st.subheader('Outreach list')
            st.caption('Lowest conversion probability first. Filter the cohort, then export the rows you want to review.')
            view = frame.copy()
            filters = st.columns(2)
            for col, field in zip(filters[:2], ['country', 'device_type']):
                if field in view:
                    choices = col.multiselect(field.replace('_', ' ').capitalize(), sorted(frame[field].dropna().unique(), key=str))
                    if choices:
                        view = view[view[field].isin(choices)]
            if st.checkbox('Only show trials below the outreach cutoff', key='candidates_only'):
                view = view[view.conversion_probability < cutoff]
            st.caption(f'{len(view):,} of {len(frame):,} trials · original cohort ranks preserved')
            if view.empty:
                st.info('No trials match these filters. Clear country or device selections, or turn off the cutoff filter.')
            st.dataframe(view, hide_index=True, width='stretch', alt='Trials ranked from lowest predicted conversion probability', column_config={
                'rank': st.column_config.NumberColumn('Rank', help='Rank in the original scored cohort; filtering preserves ranks.'),
                'trial_id': st.column_config.TextColumn('Trial ID'),
                'user_id': st.column_config.TextColumn('User ID'),
                'country': 'Country',
                'device_type': 'Device',
                'conversion_probability': st.column_config.ProgressColumn('Conversion probability', min_value=0, max_value=1, format='percent'),
                'cancellation_risk': st.column_config.ProgressColumn('Non-conversion probability', help='1 minus conversion probability. Export retains the original cancellation_risk field name.', min_value=0, max_value=1, format='percent'),
            })
            st.download_button('Download filtered ranked list', view.to_csv(index=False).encode(), f'ranked_{latest.day}.csv', 'text/csv', on_click='ignore', icon=':material/download:')

    if page == 'History':
        st.subheader('Cohort history')
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
                    except (ValueError, UnicodeError, pd.errors.ParserError, BotoCoreError, ClientError):
                        failures.append(daily[day].key)
                if day in selected:
                    _, report = selected[day]
                    row.update(drifted_share=report['drifted_share'], verdict='Drift' if report['drift'] else 'No drift',
                               source='Synthetic backfill' if report.get('source') == 'backfill' else 'Measured report')
                rows.append(row)
        table = pd.DataFrame(rows)
        numeric = ['scored_trials', 'mean_conversion', 'outreach_candidates', 'drifted_share']
        table[numeric] = table[numeric].apply(pd.to_numeric, errors='coerce')
        x, y = st.columns(2)
        x.metric('Days with predictions', f"{table.scored_trials.notna().sum()} / {len(table)}")
        measured_days = table.source.eq('Measured report')
        y.metric('Measured days with drift', f"{int((table.verdict.eq('Drift') & measured_days).sum())} / {int(measured_days.sum())}")
        st.caption('Measured drift reports take precedence over synthetic backfill on the same date. Backfilled verdicts are illustrative history, not measured checks. Missing days remain gaps.')
        x, y = st.columns(2)
        with x:
            st.markdown('**Predicted conversion over time**')
            st.line_chart(table.set_index('date')[['mean_conversion']].mul(100).rename(columns={'mean_conversion': 'Mean conversion (%)'}), color='#2563eb', alt='Daily mean predicted conversion, in percent')
            st.markdown('**Daily scoring volume**')
            st.bar_chart(table.set_index('date')[['scored_trials']].rename(columns={'scored_trials': 'Scored trials'}), color='#0f766e', alt='Number of scored trials per day')
        with y:
            st.markdown('**Share of features drifted**')
            measured = table.drifted_share.where(table.source.eq('Measured report'))
            backfill = table.drifted_share.where(table.source.eq('Synthetic backfill'))
            chart = pd.DataFrame({'date': table.date, 'Measured': measured, 'Backfill': backfill}).set_index('date')
            st.line_chart(chart.mul(100), color=['#dc2626', '#64748b'], alt='Percentage of features drifted, separated by measured and synthetic reports')
            st.caption('Share of features drifted (%). The stored verdict determines drift status; gaps mean no report.')
        st.dataframe(table, hide_index=True, width='stretch', alt='Daily prediction and monitoring history; missing values indicate unavailable data', column_config={
            'date': st.column_config.DateColumn('Date', format='MMM D, YYYY'),
            'scored_trials': st.column_config.NumberColumn('Scored trials', format='%d'),
            'mean_conversion': st.column_config.NumberColumn('Mean conversion', format='percent'),
            'outreach_candidates': st.column_config.NumberColumn('Below cutoff', format='%d'),
            'drifted_share': st.column_config.NumberColumn('Features drifted', format='percent'),
            'verdict': 'Drift verdict', 'source': 'Report source',
        })
        st.download_button('Download daily history', table.to_csv(index=False).encode(), f'history_{start}_{stop}.csv', 'text/csv', on_click='ignore', icon=':material/download:')
    if failures:
        st.warning('Some artifacts could not be read or validated; affected results are unavailable.')
        with st.expander('Affected S3 keys'):
            st.write(sorted(set(failures)))


if __name__ == '__main__':
    main()

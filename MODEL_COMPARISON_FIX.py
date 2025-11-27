# Clean Model Comparison Section
model_comparison_code = '''
# Model Comparison Page
if selected == 'Model Comparison':
    st.title("📊 Model Performance Comparison")

    st.markdown("### 🏆 Traditional ML Model Performance")

    # Load performance metrics
    try:
        # Traditional model metrics
        if os.path.exists('models/all_metrics_summary.json'):
            with open('models/all_metrics_summary.json', 'r') as f:
                traditional_metrics = json.load(f)
        else:
            # Use individual metric files
            traditional_metrics = {}
            metric_files = [
                ('diabetes', 'models/diabetes_model_metrics.json'),
                ('heart_disease', 'models/heart_disease_model_metrics.json'),
                ('parkinsons', 'models/parkinsons_model_metrics.json'),
                ('liver', 'models/liver_model_metrics.json'),
                ('hepatitis', 'models/hepititisc_model_metrics.json'),
                ('chronic_kidney', 'models/chronic_model_metrics.json')
            ]
            
            for disease, file_path in metric_files:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        traditional_metrics[disease] = json.load(f)

        # Create comparison dataframe with traditional ML models only
        comparison_data = []
        for disease in traditional_metrics.keys():
            metrics = traditional_metrics[disease]
            comparison_data.append({
                'Disease': disease.replace('_', ' ').title(),
                'Accuracy': metrics.get('accuracy', 0),
                'Precision': metrics.get('precision', 0),
                'Recall': metrics.get('recall', 0),
                'F1 Score': metrics.get('f1_score', 0)
            })

        if comparison_data:
            df = pd.DataFrame(comparison_data)

            # Display comparison table
            st.markdown("#### 📋 Model Performance Comparison Table")
            st.dataframe(df.style.format({
                'Accuracy': '{:.3f}',
                'Precision': '{:.3f}',
                'Recall': '{:.3f}',
                'F1 Score': '{:.3f}'
            }), use_container_width=True)

            # Visualization - Accuracy comparison
            st.markdown("#### 📈 Accuracy Comparison")
            fig = px.bar(
                df,
                x='Disease',
                y='Accuracy',
                title='Model Accuracy by Disease',
                color='Accuracy',
                color_continuous_scale='viridis',
                text='Accuracy'
            )
            fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig.update_layout(
                xaxis_tickangle=-45,
                yaxis=dict(range=[0, 1]),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)

            # All metrics comparison
            st.markdown("#### 📊 All Metrics Comparison")
            df_melted = df.melt(id_vars=['Disease'], var_name='Metric', value_name='Score')
            
            fig2 = px.bar(
                df_melted,
                x='Disease',
                y='Score',
                color='Metric',
                title='All Performance Metrics by Disease',
                barmode='group'
            )
            fig2.update_layout(
                xaxis_tickangle=-45,
                yaxis=dict(range=[0, 1]),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Best performing models
            st.markdown("#### 🏆 Best Performing Models")
            best_accuracy = df.loc[df['Accuracy'].idxmax()]
            best_precision = df.loc[df['Precision'].idxmax()]
            best_recall = df.loc[df['Recall'].idxmax()]
            best_f1 = df.loc[df['F1 Score'].idxmax()]

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Best Accuracy",
                    f"{best_accuracy['Disease']}",
                    f"{best_accuracy['Accuracy']:.3f}"
                )
            with col2:
                st.metric(
                    "Best Precision",
                    f"{best_precision['Disease']}",
                    f"{best_precision['Precision']:.3f}"
                )
            with col3:
                st.metric(
                    "Best Recall",
                    f"{best_recall['Disease']}",
                    f"{best_recall['Recall']:.3f}"
                )
            with col4:
                st.metric(
                    "Best F1 Score",
                    f"{best_f1['Disease']}",
                    f"{best_f1['F1 Score']:.3f}"
                )
        else:
            st.info("No performance data available for comparison.")

    except Exception as e:
        st.error(f"❌ Error loading performance metrics: {e}")
        st.info("""
        💡 **Tip:** Model metrics files are missing. 
        
        To see model comparisons, ensure these files exist in the models/ folder:
        - diabetes_model_metrics.json
        - heart_disease_model_metrics.json
        - parkinsons_model_metrics.json
        - liver_model_metrics.json
        - hepititisc_model_metrics.json
        - chronic_model_metrics.json
        
        Or create models/all_metrics_summary.json with all metrics combined.
        """)
'''

print("Model Comparison section code generated successfully!")
print("This code should replace lines 2417-2600 in app.py")

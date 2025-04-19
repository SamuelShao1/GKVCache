import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Read the CSV files
global_df = pd.read_csv('trivia_global.csv')
user_df = pd.read_csv('trivia_user.csv')

# Add a column to identify the cache strategy
global_df['strategy'] = 'Global KV Cache'
user_df['strategy'] = 'User KV Cache'

# Combine the dataframes
combined_df = pd.concat([global_df, user_df], ignore_index=True)

# Convert timestamp to datetime for potential time series analysis
combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])

# Set a beautiful aesthetic style and color palette
plt.style.use('fivethirtyeight')
custom_colors = ["#4E79A7", "#F28E2B"]  # Professional blue and orange

# Create figure with improved proportions and resolution
fig = plt.figure(figsize=(18, 15), dpi=120)
fig.patch.set_facecolor('#F8F9FA')  # Light background for the entire figure

# Create a 2x2 grid with proper spacing (adjusted since the 5th plot is removed)
gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)

# Add a title to the entire figure
fig.suptitle('Cache Strategy Performance Comparison', 
             fontsize=22, fontweight='bold', y=0.98, 
             fontfamily='DejaVu Sans')

# Function to style the axes consistently
def style_axis(ax, title):
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15, fontfamily='DejaVu Sans')
    ax.set_xlabel('Cache Strategy', fontsize=14, fontweight='medium', labelpad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_facecolor('#F0F0F5')  # Subtle background

    # Add value labels on top of bars
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height):
            ax.text(p.get_x() + p.get_width() / 2.,
                    height + (height * 0.02),
                    f'{height:.2f}' if height < 100 else f'{height:.1f}',
                    ha="center", fontsize=12, fontweight='bold')

# 1. Inference Time Plot - Top Left
ax1 = fig.add_subplot(gs[0, 0])
bars1 = sns.barplot(x='strategy', y='processing_time_ms', data=combined_df, 
                    errorbar=('ci', 95), capsize=0.2, palette=custom_colors, ax=ax1)
style_axis(ax1, 'Inference Time')
ax1.set_ylabel('Processing Time (ms)', fontsize=14, fontweight='medium', labelpad=10)

# 2. Memory Usage Plot - Top Right
ax2 = fig.add_subplot(gs[0, 1])
bars2 = sns.barplot(x='strategy', y='memory_usage_mb', data=combined_df, 
                   errorbar=('ci', 95), capsize=0.2, palette=custom_colors, ax=ax2)
style_axis(ax2, 'Memory Usage')
ax2.set_ylabel('Memory Usage (MB)', fontsize=14, fontweight='medium', labelpad=10)

# 3. Semantic Search Time Plot - Bottom Left
ax3 = fig.add_subplot(gs[1, 0])
bars3 = sns.barplot(x='strategy', y='similarity_search_time_ms', data=combined_df, 
                   errorbar=('ci', 95), capsize=0.2, palette=custom_colors, ax=ax3)
style_axis(ax3, 'Semantic Search Time')
ax3.set_ylabel('Similarity Search Time (ms)', fontsize=14, fontweight='medium', labelpad=10)

# 4. Cache Hit Rate Plot - Bottom Right (replacing Similarity Score)
ax4 = fig.add_subplot(gs[1, 1])
if 'cache_hit' in combined_df.columns:
    combined_df['cache_hit_numeric'] = combined_df['cache_hit'].map({True: 1, False: 0})
    hit_rates = combined_df.groupby('strategy')['cache_hit_numeric'].mean() * 100
    hit_rate_df = pd.DataFrame({'strategy': hit_rates.index, 'hit_rate': hit_rates.values})
    bars4 = sns.barplot(x='strategy', y='hit_rate', data=hit_rate_df,
                       palette=custom_colors, ax=ax4)
    style_axis(ax4, 'Cache Hit Rate')
    ax4.set_ylabel('Hit Rate (%)', fontsize=14, fontweight='medium', labelpad=10)
    ax4.set_ylim([0, 100])
elif 'cumulative_hit_rate' in combined_df.columns:
    last_records = combined_df.sort_values('timestamp').groupby('strategy').last()
    hit_rate_df = pd.DataFrame({
        'strategy': last_records.index,
        'hit_rate': last_records['cumulative_hit_rate'].values * 100
    })
    bars4 = sns.barplot(x='strategy', y='hit_rate', data=hit_rate_df,
                       palette=custom_colors, ax=ax4)
    style_axis(ax4, 'Cumulative Cache Hit Rate')
    ax4.set_ylabel('Hit Rate (%)', fontsize=14, fontweight='medium', labelpad=10)
    ax4.set_ylim([0, 100])
else:
    ax4.text(0.5, 0.5, 'No cache hit rate data available', 
             ha='center', va='center', fontsize=14)
    ax4.set_title('Cache Hit Rate', fontsize=16, fontweight='bold', pad=15)

# Add a subtle watermark
fig.text(0.95, 0.05, 'Cache Analysis', fontsize=12, color='gray', 
         ha='right', va='bottom', alpha=0.5, rotation=0)

# Add explanatory footnote
fig.text(0.1, 0.02, 'Note: Error bars represent 95% confidence intervals', 
         fontsize=10, style='italic', color='#555555')

# Calculate summary statistics for the text box
global_stats = combined_df[combined_df['strategy'] == 'Global KV Cache']
user_stats = combined_df[combined_df['strategy'] == 'User KV Cache']
proc_diff = global_stats['processing_time_ms'].mean() - user_stats['processing_time_ms'].mean()
mem_diff = global_stats['memory_usage_mb'].mean() - user_stats['memory_usage_mb'].mean()
search_diff = global_stats['similarity_search_time_ms'].mean() - user_stats['similarity_search_time_ms'].mean()

# Add text box with key insights
insight_text = (
    f"Key Insights:\n"
    f"• Processing Time: {abs(proc_diff):.1f}ms {'higher' if proc_diff > 0 else 'lower'} with Global Cache\n"
    f"• Memory Usage: {abs(mem_diff):.2f}MB {'higher' if mem_diff > 0 else 'lower'} with Global Cache\n"
    f"• Search Time: {abs(search_diff):.2f}ms {'higher' if search_diff > 0 else 'lower'} with Global Cache"
)
fig.text(0.74, 0.18, insight_text, fontsize=12,
         bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5',
                  edgecolor='#CCCCCC'))

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for the title
plt.savefig('cache_strategy_comparison_enhanced.png', dpi=300, bbox_inches='tight')
plt.show()

# Calculate detailed summary statistics
summary_stats = combined_df.groupby('strategy').agg({
    'processing_time_ms': ['mean', 'std', 'min', 'max', 'count'],
    'memory_usage_mb': ['mean', 'std', 'min', 'max'],
    'similarity_search_time_ms': ['mean', 'std', 'min', 'max'],
    'similarity_score': ['mean', 'std', 'min', 'max', 'count']  # Kept for reporting only
})

print("Summary Statistics:")
print(summary_stats)

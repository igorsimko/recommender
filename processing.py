

def get_proceed_data(activity, deal_items, deal_details):
    activity_items = activity.merge(deal_items, left_on='dealitem_id', right_on='id', how='outer')
    full_data = activity_items.merge(deal_details, left_on='deal_id_x', right_on='id', how='outer')

    grouped_by_users = full_data.groupby('user_id')
    grouped_by_dealitem_id = full_data.groupby('dealitem_id')

    return full_data, grouped_by_users, grouped_by_dealitem_id